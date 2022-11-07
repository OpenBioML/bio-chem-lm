import math
from collections import namedtuple
from functools import reduce

import torch
import torch.nn.functional as F
from torch import nn
from bio_lm.model.pretrained import ElectraPreTrainedModel

# copied from lucidrains and updated
# constants

Results = namedtuple(
    "Results",
    [
        "loss",
        "mlm_loss",
        "disc_loss",
        "gen_acc",
        "disc_acc",
        "disc_labels",
        "disc_predictions",
    ],
)

# helpers


def log(t, eps=1e-9):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0):
    # don't need to do softmax since logits will be in right relative
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)
    # count how many valid tokens there are (excluding pad/cls/special)
    num_tokens = mask.sum(dim=-1, keepdim=True)

    # mask tokens if pos > prob * num_valid token
    mask_excess = mask.cumsum(dim=-1) > (num_tokens * prob).ceil()
    mask_excess = mask_excess[:, :max_masked]

    # gen random numbers but mask non-valid tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)

    # get top N (max_masked) from random indices
    # shuffled since randn
    _, sampled_indices = rand.topk(max_masked, dim=-1)

    # mask valid tokens -> set to 0
    # why do we add 1? -> https://github.com/lucidrains/electra-pytorch/issues/5
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


# hidden layer extractor class, for magically adding adapter to language model to be pretrained


class HiddenLayerExtractor(nn.Module):
    def __init__(self, net, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden


# main electra class


class Electra(ElectraPreTrainedModel):
    def __init__(
        self,
        generator,
        discriminator,
        *,
        config=None,
        num_tokens=None,
        discr_dim=-1,
        discr_layer=-1,
        mask_prob=0.15,
        replace_prob=0.85,
        random_token_prob=0.0,
        mask_token_id=2,
        pad_token_id=0,
        mask_ignore_token_ids=[],
        disc_weight=50.0,
        gen_weight=1.0,
        temperature=1.0,
    ):
        super().__init__(config=config)

        self.generator = generator
        self.discriminator = discriminator

        if discr_dim > 0:
            self.discriminator = nn.Sequential(
                HiddenLayerExtractor(discriminator, layer=discr_layer),
                nn.Linear(discr_dim, 1),
            )

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight
        self.gen_weight = gen_weight

        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        gen_labels = input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        replace_prob = torch.full(labels.shape, 1 - self.replace_prob).to(input_ids.device)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(input_ids, self.mask_ignore_token_ids).to(input_ids.device)
        replace_prob.masked_fill_(no_mask, value=0.0)
        masked_indices = torch.bernoulli(replace_prob).bool().to(input_ids.device)
        gen_labels[~masked_indices] = -100
        
        masked_input = input_ids.clone()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(input_ids.device).bool() & masked_indices
        masked_input[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(input_ids.device).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.config.vocab_size, gen_labels.shape, dtype=torch.long).to(input_ids.device)
        masked_input[indices_random] = random_words[indices_random] 

        # get generator output and get mlm loss
        logits = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        mlm_loss = logits.loss
        logits = logits.logits
        
        # use mask from before to select logits that need sampling
        # select tokens that could be potentially changed from mlm objective
        sample_logits = logits[masked_indices]

        # # sample
        sampled = gumbel_sample(sample_logits, temperature=self.temperature)

        # # scatter the sampled values back to the input
        disc_input = input_ids.clone()
        disc_input[masked_indices] = sampled.detach()

        # generate discriminator labels, with replaced as True and original as False
        disc_labels = (input_ids != disc_input).float().detach()

        # get discriminator predictions of replaced / original
        non_padded_indices = torch.nonzero(
            input_ids != self.pad_token_id, as_tuple=True
        )

        # get discriminator output and binary cross entropy loss
        disc_logits = self.discriminator(
            disc_input,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        disc_logits = disc_logits.logits.reshape_as(disc_labels)

        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices], disc_labels[non_padded_indices]
        )

        # gather metrics
        with torch.no_grad():
            gen_predictions = torch.argmax(logits, dim=-1)
            disc_predictions = torch.round((torch.sign(disc_logits) + 1.0) * 0.5)
            gen_acc = (gen_labels[masked_indices] == gen_predictions[masked_indices]).float().mean()
            disc_acc = (
                0.5 * (disc_labels[masked_indices] == disc_predictions[masked_indices]).float().mean()
                + 0.5 * (disc_labels[~masked_indices] == disc_predictions[~masked_indices]).float().mean()
            )

        # return weighted sum of losses
        return Results(
            self.gen_weight * mlm_loss + self.disc_weight * disc_loss,
            mlm_loss,
            disc_loss,
            gen_acc,
            disc_acc,
            disc_labels,
            disc_predictions,
        )._asdict()
