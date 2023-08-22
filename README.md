# tiny-py-transformer
Small transformer, weekend project, implementation using pytorch, without using `torch.nn.Transformer`, that can ramble in a shakespearian style.

## Output example

The example bellow is produced by a model with, 2 decoder layers, block and batch size of 16, embedding size of 64 and a head size of 32.

```text
Loss: 0.32 - Val loss: 0.39 - 20%
--- Generating ---
.
CHHHHHHHHHHHHHHAs and spict,
You shouse so agains besto his but I eyur, I should some le which so for the bearss all spos made sens,
That of grace Med exulsters the dis an with the ke with cannot mattles, will le woman dange fill somesly;
My sith noble,
Ser hearts some some, the troly sto but stady paunse. my Bcr
The: the untong strens of my man shourse the sucng rones.

Yjes breasters will the deruse yourse go man by the lord's ca;
What some the refaols your in brothe king disumd the brother he, Rome the founse some so, the come to boson of betters you spil the knay brothe secon?

ESTON:
But a bosome.
Son't dishos you the word
The ke in to constake of scogent:
Thys comest the uns and so to the hanstrenss mother he ried crown;
This I cannot with me so 
```

**Example of text, with individual tokens separated by `|`:**

```text
ss: 0.45 - Val loss: 0.50 - 0%
--- Generating ---
|.|
|'|A| |fl|s|ti|ya|s| |thi| |thi|s| |fu|l| |pro|me| |kin|d| |s|to| |all| |s|ti|w| |to| |the| |the| |cha|nce|;| |he| |th|li|pe|s|,| |of| |you| |dea|h| |to| |s|on| |of| |s|to| |I| |I| |s|en|li|s|e|s|,| |kin|g| |all|ow| |in| |ag|rc| |lor|d|s| |s|s|in|
|I| |hou|t| |s|ee|m|
|
|po|or| |and| |I| |M|y| |s|o| |s|end|t|s| |fa|ge|l| |my| |s|pe|ll|s| |our| |ho|the| |gi|re|s|
|A| |lov|ugh|iu| |of| |am|e| |the| |s|ca|ll| |tha|e| |my| |ma|le|s| |en|uc|e| |pa|rc| |u|s| |s|cu|.|
|
|N|U|K|N|E|R|:|
|T|her|e| |a| |the| |you| |the| |you| |me| |the| |s|on| |a| |for|,| |tho|u| |you| |s|ome| |pr|ld| |har|s|t| |s|ome| |you| |to| |le|.|
|
|A|U|T|le|ing| |hat|h| |of| |thi|s| |s|er|vi|le|.|
|T|I|:|
|I| |wil|l| |'|d| |dea|d| |you|s|t| |M|e| |on| |are| |ho|no| |the| |hea|r| |you| |s|et| |me| |the| |s|hal|l| |bo| |her| |the|m| |ld|e| |wil|l| |the| |my| |be| |the| |fi|y| |you|r| |of| |you| |s|el|f|;| |he| |hav|e| |of| |my| |men|d| |wi|s|s|de|pe|y|
|W|hat| |do| |my| |I| |s|ir| |the|m| |the|r| |him| |s|per|s|t|.|
|
|A|s| |tr|e|s| |s|o| |s|pi|d|,|
|I| |s|tr|e| |I| |s|s|co|ee|,| |fa|l| |wor|d|s|s| |a| |un|s| |ye|eat| |s|ee|ge| |s|ill|,|
|O| |po|or| |a| |re|mi|;| |the| |my| |s|hal|l|,|
|T|om| |a| |him| |if| |all| |be| |s|we|d| |thi|nk| |s|tr|ck|,|
|I|f| |s|wer
```

## Thanks a lot to...
[Andrej Karpathy](https://github.com/karpathy), for making a [video where he implements transformers with pytorch](https://yewtu.be/watch?v=kCc8FmEb1nY), and for publishing his code in [a git repo](https://github.com/karpathy/ng-video-lecture).

For making sure I really understood all the concepts in the video, I made sure not to write anything off (expect the code in data.py). Instead I opted to watch segments of the video, then trying to implement it on my own, to make sure I really absorbed all the concepts.
