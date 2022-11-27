# Electra Model 

We copy and paste the code from HuggingFace as we need to make some changes to the code.
We have added the following things to the code 

- Adding different types of layer normalizations
- Allowing for training with [Maximal Update Parameterization](https://github.com/microsoft/mup)
- Added support for other positional encodings such as ALiBi and Rotary