# HUMAC: Enabling Multi-Robot Collaboration from Single-Human Guidance
[Zhengran Ji](https://jzr01.github.io/)¹, [Lingyu Zhang](https://lingyu98.github.io/)¹, [Paul Sajda](https://liinc.bme.columbia.edu/people/paul-sajda)², [Boyuan Chen](http://boyuanchen.com/)¹

¹ Duke University, ² Columbia University

[website](http://www.generalroboticslab.com/HUMAC) | [paper]() | [video]()

![Multi-Agent/Robot Collaboration](images/Teaser.jpeg)


## Overview
Learning collaborative behaviors is essential for multi-agent systems. Traditionally, multi-agent reinforcement learning solves this implicitly through a joint reward and centralized observations, assuming collaborative behavior will emerge. Other studies propose to learn from demonstrations of a group of collaborative experts. Instead, we propose an efficient and explicit way of learning collaborative behaviors in multi-agent systems by leveraging expertise from only a single human. Our insight is that humans can naturally take on various roles in a team. We show that agents can effectively learn to collaborate by allowing a human operator to dynamically switch between controlling agents for a short period and incorporating a human-like theory-of-mind model of teammates. Our experiments showed that our method improves the success rate of a challenging collaborative hide-and-seek task by up to 58% with only 40 minutes of human guidance. We further demonstrate our findings transfer to the real world by conducting multi-robot experiments.

![Method](images/Mainfig.jpeg)

## Result
### Simulation Success Rate (%)
![Method](Simulation.png)

### Real-World Experiment Success Rate (%)
![Method](real.png)


## Quick Start

1. Clone the repository:

    ```bash
    git clone https://github.com/generalroboticslab/HUMAC.git
    ```
2. To run the simulation part of the paper, install [CREW](https://github.com/generalroboticslab/CREW). There are more detailed instruction in the [Simulation](https://github.com/generalroboticslab/HUMAC/tree/main/Simulation) folder.

3. To run the real-world experiment part of the paper, navigate to [Real-World](https://github.com/generalroboticslab/HUMAC/tree/main/Real-World) folder for detailed instructiom.

## Repository Structure
This repository has this following structure
```plaintext
├── Simulation              
│   └── crew-algorithms
│   └── environment
│   └── training
├── Real-World
│   └── environment
│   └── training
├── images
├── .gitignore              
├── README.md           
└── LICENSE             

```

## Acknowledgement

```plaintext
    This work is supported by ARL STRONG program under awards W911NF2320182 and W911NF2220113. We would also like to thank Jiaxun Liu for helping with the hardware setup.
```

## Citation

If you think this paper is helpful, please consider cite our work

    ```bash
    git clone https://github.com/generalroboticslab/HUMAC.git
    ```

