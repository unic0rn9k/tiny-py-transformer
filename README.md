# tiny-py-transformer
Small transformer implementation using pytorch

Everyone kept punking me about learning python and pytorch,
so here is a proof of concept transformer implementation...

## Output example

The example bellow is produced by a model with, 2 decoder layers, block and batch size of 16, embedding size of 64 and a head size of 32.

```text
Loss: 0.78 - Val loss: 0.84 - 31%
--- Generating ---
. I'
Oe IY i akn hiv ininghaler'd phondoren,
hellore,
Upy mingeain,

Ot thele rade hour loire,
I norer'le is flire
I mourirringhe chou nont ingloray, vishot the, larenenjerle,
SINGORIO:
An'e,

LIODIDINTheard, a fhou slarle,
Phore of sair whith.

CIKINGERIONT:
Ond thy nohed,
my lak ingan'ens'll had.

Lhimims han han'lalle,
U Bhe, I nore ond shancorkinghene, unore, dheare foigmhangen of orere

Leardrear, e ne,

LADYDNIG:

delywlayunothle
Thair', inllolles
```

## Thanks a lot to...
[Andrej Karpathy](https://github.com/karpathy), for making a [video where he implements transformers with pytorch](https://yewtu.be/watch?v=kCc8FmEb1nY). And for publishing his code in [a git repo](https://github.com/karpathy/ng-video-lecture).

For making sure I really understood all the concepts in the video, I made sure not to write anything off (expect the code in data.py). Instead I opted to watch segments of the video, then trying to implement it on my own, to make sure I really absorbed all the concepts.