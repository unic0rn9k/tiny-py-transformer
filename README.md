# tiny-py-transformer
Small transformer, weekend project, implementation using pytorch, without using `torch.nn.Transformer`, that can ramble in a shakespearian style.

## Output example

The example bellow is produced by a model with, 2 decoder layers, block and batch size of 16, embedding size of 64 and a head size of 32.

```text
Loss: 0.24 - Val loss: 0.26 - 27%
--- Generating ---
.

CCCCCCCCCCCCCUUKE:
I I Richard:
I more, my will's ance my a be that the gentle my ENIUS:
BERUK:
Hhere:
But hat in or of in thees.

Lord Isir of have streem to shall see a abue,
But tost have and days to straim'd and there cheer sincion.

LUCLUS:
A las may to how of of thived of theres putt harm him my shal Marcius and death,
Plast is I hear bas your I kind with have to death:
For to to most in her head to my the the to is pasty all apriin, we ne your will for have naon
I NINIUS:
Hath we loved I wobow.

PUCHUDUKE:
The say is his king.

BUCUS:
Nay you, am morew, this brother since.

And to thine this lifes heress that sinceoiees ans;
To with to my lord Ry no am hiss here the dearst the stired the do her Marcience
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
