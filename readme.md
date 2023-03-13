# Provide a memory record package

refer from [it](https://www.sicara.fr/blog-technique/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks)

run `python mem_record.py >./log.txt` 
in log.txt u can see the details.

It is a way to learn how to use hook in pytorch
It is also a way to learn how to record the memory use in every layer in model when we are training it in GPU.


A few things to observe:

The memory keeps increasing during the forward pass and then starts decreasing during the backward pass
The slope is pretty steep at the beginning and then flattens: