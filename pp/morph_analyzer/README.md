# LookupAnalyzerDisambiguator

### Introduction 
A tool for Turkish language processing which inputs Turkish tokens (text tokenized into tokens such as words and punctuations) and outputs their disambiguated morphological analyzes.
Since Turkish has highly rich morphology, morphological analysis is required for many tasks including pos-tagging, dependency parsing


In our solution, we implement a simple morhological analyzer based on stem and suffix dictionaries.
Using this simple morphological analzyzer, all possible analyzes of each token is generated.
A neural network (specifically Bidirectional character based LSTMs) is implemented using DyNet library and trained for selecting the correct morphological analysis among all possible analyzes according to the context which words have appeared in.
The neural network architecture is similar to the architecture used in [Shen et. al.'s study](http://www.aclweb.org/anthology/C16-1018)

### Performance

Although we do not use any complex morphological analyzer as in most of the studies,
our results are competitive with state-of-the-art morphological disambiguators (96%~97% accuracy).

We will report a comprehensive evaluation results soon.

### Usage

Just build a docker image using Dockerfile in the repo:

```commandline 
docker build --tag turkish-tagger .
```

Then start a docker container using the docker image built on previos step:

```commandline
docker run -p 8081:8081 -d  turkish-tagger
```

Then, it will start to serve as a  web application if everything goes well.
You can just send a post request to `localhost:8081/analyze` to analyze a Turkish sentence morphologically.

#### Example requests and responses

**Request1 :**
```cURL 
curl --request POST \
  --url http://localhost:8081/analyze \
  --header 'cache-control: no-cache' \
  --header 'content-type: application/json' \
  --header 'postman-token: c18af364-c1cb-cc41-0903-063547ac7fce' \
  --data '{
    "tokens" : [
        "alın",
        "yazısı"
    ]}'
```

**Response 1:**
```json
[
    "alın+Noun+A3sg+Pnon+Nom",
    "yazı+Noun+A3sg+P3sg+Nom"
]
```

**Request2 :**
```cURL 
curl --request POST \
  --url http://localhost:8081/analyze \
  --header 'cache-control: no-cache' \
  --header 'content-type: application/json' \
  --header 'postman-token: dd9b686d-509c-d676-173c-f8f64d5dcee0' \
  --data '{
    "tokens" : [
        "gelirken",
        "ekmek",
        "alın",
        "."
     ]}'
```

**Response 2:**
```json
[
    "gelir+Noun+A3sg+Pnon+Nom^DB+Verb+Zero^DB+Adverb+While",
    "ekmek+Noun+A3sg+Pnon+Nom",
    "al+Verb+Pos+Imp+A2pl",
    ".+Punc"
]
```


#### Notes

Please email me and ask for permission to use this tool.
Also note that this is not a release version and may contain some bugs.
Every contribution is welcome.

We still continue working with my advisor in my PhD thesis. Wait for better accuracies :)


