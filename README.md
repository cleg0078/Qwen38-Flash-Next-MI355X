# Qwen38-Flash-Next-MI355X
How to run Qwen 3.8 Flash Next on a single MI355Xi?

I wanted to test a very high end GPU compared a 2 nodes DGX Spark cluster.

It was not out of the box. 

I rented a GPU droplet from DIGITAL OCEAN for 4.5$/hour. 

It's the FP8 model, MTP activated. 

I tried to run the mxfp4 model but needs a lot of work. I stopped after 4 hours. The script will be added maybe later. I need to prepare more. I had to disable all hardware gpu acceleration. It loads but only 20tok/s. 


For the FP8, I reached for 1 stream between 270 and 180 tokens/s. I will use it more for real projects. 4.5$/hour must be well used! 

Use this script as a template. I know it's probably not the best but it works... and it was the target. 

Have fun. 


