---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* 2024.09 - now, Ph.D in Computer Science, School of Artificial Intelligence, Xidian University
* 2020.09 - 2024.06, B.Eng. of Artificial Intelligence, School of Artificial Intelligence, Xidian University
* 2017.09 - 2020.07,  Suining No.1 High School

Internship and Work experience
======
no work experience

Honors and Awards
======

  <ul>{% for post in site.haa reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Publications
======

  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
Talks
======

  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
Teaching
======

  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
