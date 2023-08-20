# tiny-py-transformer
Small transformer implementation using pytorch, without using `torch.nn.Transformer`.

## Output example

The example bellow is produced by a model with, 2 decoder layers, block and batch size of 16, embedding size of 64 and a head size of 32.

```text
Loss: 0.45 - Val loss: 0.46 - 99%
--- Generating ---
. o od o o a o ied
HIN:
Iut heser yourt ropig youd willimor fourertheisiederayou boust blin ondill aug appearde bladouperiefeands wan bne.

ou ll hirsiem yougdien whirsougebly,
And bende wientere hit and abat win wieg indue hand are po wiice
Lieiceing nereyd wice our; our?

IIU:
Id, hitiedit wae ming towiot fries thind eond wine pourdis meer ho'ding binge hor out ince quicuernt thouysiou dour. lourn.

IUS:
This and haie 'ed
ARI thoroous oering ou miou oug ald oukeremeris wils ale re yousucieint th
```

## Thanks a lot to...
[Andrej Karpathy](https://github.com/karpathy), for making a [video where he implements transformers with pytorch](https://yewtu.be/watch?v=kCc8FmEb1nY). And for publishing his code in [a git repo](https://github.com/karpathy/ng-video-lecture).

For making sure I really understood all the concepts in the video, I made sure not to write anything off (expect the code in data.py). Instead I opted to watch segments of the video, then trying to implement it on my own, to make sure I really absorbed all the concepts.
