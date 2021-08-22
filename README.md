# DD-GAN

## Domain Decomposition Predictive Generative Adversarial Network for modelling fluid flow

[![codecov](https://codecov.io/gh/acse-jat20/DD-GAN/branch/main/graph/badge.svg?token=1LU7UG5OF9)](https://codecov.io/gh/acse-jat20/DD-GAN)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/acse-jat20/DD-GAN/blob/main/LICENSE)
![example workflow](https://github.com/acse-jat20/DD-GAN/actions/workflows/health.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- [![Documentation Status](https://github.com/acse-jat20/DD-GAN/actions/workflows/docs.yml/badge.svg)](https://github.com/acse-jat20/DD-GAN/blob/main/docs/docs.pdf) -->

<!-- PROJECT LOGO -->

<br />
<p align="center">
  <a href="https://github.com/acse-jat20/DD-GAN">
    <img src="images/flowcharts/GAN2.png" alt="Logo" width="785" height="538">
  </a>

<p align="center">
    <br />
    <a href="https://github.com/acse-jat20/DD-GAN/blob/main/docs/docs.pdf"><strong>Explore the docs»</strong></a>
    <br />
    <br />
    <a href="https://github.com/acse-jat20/DD-GAN/issues">Report Bug</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This project contains an intuitive library for interacting with a domain decomposition predictive GAN. This draws on ideas from recent research on [domain decomposition methods for reduced order modelling](https://www.sciencedirect.com/science/article/pii/S0045793019300350) and [predictive GANs](https://arxiv.org/abs/2105.07729). It also contains some test examples. 

<!-- GETTING STARTED -->

## Prerequisites

* Python 3.8
* Tensorflow and other packages in ```requirements.txt```

## Getting Started

### Installation


1. ```git clone https://github.com/acse-jat20/DD-GAN```
2. ```cd ./DD-GAN```
3. ```pip install -e .```

<!-- USAGE EXAMPLES -->

### Usage

In a python file, import the following to use all of the functions:

```python
import ddgan
```

### Example data:

* The POD coefficients used in the project can be found under [/data](./data/processed/DD/) *- Original .vtu files curteousy of Dr. Claire Heaney*

<!-- ACKNOWLEDGEMENTS 
_For more information, please refer to the report in this repo_
-->
<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->


## Contact

* Tómasson, Jón Atli jon.tomasson1@gmail.com

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

* Dr. Claire Heaney
* Prof. Christopher Pain
* Zef Wolffs

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links 
[contributors-shield]: https://img.shields.io/github/contributors/acse-2020/group-project-the-uploaders.svg?style=for-the-badge
[contributors-url]: https://github.com/acse-2020/acse-4-x-ray-classification-losslandscape/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/acse-2020/group-project-the-uploaders.svg?style=for-the-badge
[issues-url]: https://github.com/acse-2020/acse-4-x-ray-classification-losslandscape/issues
[license-shield]: https://img.shields.io/github/license/acse-2020/group-project-the-uploaders.svg?style=for-the-badge
[license-url]: https://github.com/acse-2020/acse-4-x-ray-classification-losslandscape/blob/main/LICENSE.txt
-->
