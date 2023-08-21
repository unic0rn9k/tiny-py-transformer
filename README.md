# tiny-py-transformer
Small transformer implementation using pytorch, without using `torch.nn.Transformer`.

## Output example

The example bellow is produced by a model with, 2 decoder layers, block and batch size of 16, embedding size of 64 and a head size of 32.

```text
ss: 0.62 - Val loss: 0.74 - 0%
--- Generating ---
.

LowU roddsue de wer shers star.

RIOUCI he tal on lace sos ther his histads,
I stro of brhanings taker I the hod ho is son sometalsirs that, your, whet you sir the Welf mu gooF with wiss thi stal, that he sir the it ther he spesstad shou, their stace of your stasstad.

TETER:
MBut you  Willtise sir,
Ther sendsst.
A of the, wher, sir ther, boire the shalth hand mor stain the sir strue.
We stal you stant place disidl staintatiod Fge,
Of sirse not this this I the to the the more shoue thi sons soul
I sirs strustal unat thesir she he sher you foe, tace,
I wills hiforl this tay stal,
O, sir make -more the king sir wifuls
With
MERY:
ORILI he staf stal it son sir you have selve have swer the stae she fair ther, 
Lordt 
```

## Thanks a lot to...
[Andrej Karpathy](https://github.com/karpathy), for making a [video where he implements transformers with pytorch](https://yewtu.be/watch?v=kCc8FmEb1nY), and for publishing his code in [a git repo](https://github.com/karpathy/ng-video-lecture).

For making sure I really understood all the concepts in the video, I made sure not to write anything off (expect the code in data.py). Instead I opted to watch segments of the video, then trying to implement it on my own, to make sure I really absorbed all the concepts.
