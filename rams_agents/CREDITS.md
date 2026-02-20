# Credits and Attribution

This RAMS (Reliability, Availability, Maintainability, Safety) framework integrates multiple open-source datasets, models, and research contributions.

## RL-Based Collision Avoidance

### Pre-trained PPO Policy

The RL collision avoidance policy is adapted from:

**Repository**: [Acmece/rl-collision-avoidance](https://github.com/Acmece/rl-collision-avoidance)

**Paper**: "Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning"  
- **Authors**: Long, P., Fan, T., Liao, X., Liu, W., Zhang, H., & Pan, J.
- **arXiv**: [1709.10082](https://arxiv.org/abs/1709.10082)
- **Year**: 2017

**Implementation by**: Tianyu Liu (2018)

**License**: MIT License

**Citation**:
```bibtex
@misc{Tianyu2018,
    author = {Tianyu Liu},
    title = {Robot Collision Avoidance via Deep Reinforcement Learning},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/Acmece/rl-collision-avoidance}},
    commit = {7bc682403cb9a327377481be1f110debc16babbd}
}

@article{long2018towards,
    title={Towards Optimally Decentralized Multi-Robot Collision Avoidance via 
           Deep Reinforcement Learning},
    author={Long, Pinxin and Fan, Tingxiang and Liao, Xinyi and Liu, Wenxi and 
            Zhang, Hao and Pan, Jia},
    journal={arXiv preprint arXiv:1709.10082},
    year={2017}
}
```

### Adaptation Notes

The original policy was designed for ground robots with LiDAR sensors. For maritime vessel application, we:
1. Synthesize a "virtual LiDAR scan" from Kalman-filtered track positions
2. Apply a Potential Safety Function (PSF) layer for formal safety guarantees
3. Map RL actions to COLREGS-compliant avoidance maneuvers

---

## Sensor Fusion Dataset

### Autoferry Sensor Fusion Benchmark

**Repository**: [Autoferry/sensor_fusion_dataset](https://github.com/Autoferry/sensor_fusion_dataset)

**Description**: Multi-sensor tracking dataset collected from milliAmpere autonomous ferry, with Vessel as a tracked target in some scenarios.

**Sensors**: Radar, LiDAR, EO Camera, IR Camera

**License**: MIT License

**Project**: [Autoferry Project](https://autoferry.github.io/)

**Citation**:
```bibtex
@misc{autoferry_dataset,
    author = {Autoferry Project},
    title = {Sensor Fusion Dataset},
    year = {2021},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/Autoferry/sensor_fusion_dataset}}
}
```

---

## Vessel Dynamics Reference

### Python Vehicle Simulator

**Repository**: [cybergalactic/PythonVehicleSimulator](https://github.com/cybergalactic/PythonVehicleSimulator)

**Author**: Thor I. Fossen

**Book**: "Handbook of Marine Craft Hydrodynamics and Motion Control", 2nd Edition (2021), Wiley

**License**: MIT License

**URL**: [python.fossen.biz](https://python.fossen.biz/)

---

## UCI Naval Propulsion Dataset

**Repository**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants)

**Description**: Propulsion plant degradation simulation for predictive maintenance.

**Citation**:
```bibtex
@misc{uci_naval,
    author = {Coraddu, A. and Oneto, L. and Ghio, A. and Savio, S. and 
              Anguita, D. and Figari, M.},
    title = {Condition Based Maintenance of Naval Propulsion Plants Dataset},
    year = {2016},
    publisher = {UCI Machine Learning Repository}
}
```

---

## Research Vessel

### Vessel

**Owner**: NTNU (Norwegian University of Science and Technology)

**MMSI**: 258342000 | **IMO**: 9371361

**URL**: [www.ntnu.edu/gunnerus](https://www.ntnu.edu/gunnerus)

---

## Open Simulation Platform

**URL**: [opensimulationplatform.com](https://opensimulationplatform.com)

**Vessel Models**: DP control, path-following, vessel dynamics (FMU format)

---

## Software Dependencies

- **PyTorch**: Deep learning framework ([pytorch.org](https://pytorch.org))
- **NumPy**: Numerical computing ([numpy.org](https://numpy.org))
- **Matplotlib**: Visualization ([matplotlib.org](https://matplotlib.org))
- **scikit-learn**: Machine learning ([scikit-learn.org](https://scikit-learn.org))

---

## License

This RAMS framework is released under the MIT License for research and educational use.

---

*Last updated: February 2026*
