---
title: "LoopExpose: An Unsupervised Framework for Arbitrary-Length Exposure Correction"
collection: publications
permalink: /publication/loopexpose/
order: 1
status: "Accepted"
status_key: "accepted"
venue: "IEEE Transactions on Image Processing (TIP)"
image: "/images/publications/loopexpose.png"
image_alt: "LoopExpose nested correction and multi-exposure fusion optimization framework"
excerpt: >-
  LoopExpose learns arbitrary-length exposure correction without paired labels through a nested correction-fusion loop. Multi-exposure fusion iteratively refines pseudo-labels, while luminance-ranking supervision preserves the relative brightness order of the input sequence.
codeurl: "https://github.com/FALALAS/LoopExpose"
---

<figure class="publication-detail-image">
  <img src="{{ page.image | relative_url }}" alt="{{ page.image_alt | escape }}">
</figure>

{{ page.excerpt | markdownify }}

[View code]({{ page.codeurl }})
