# Leaf-energy-balance Simulator
## Overview
*Masterthesis:* Ecophysiological adaptations of plant thermoregulation to soil and climatic conditions along a vegetation gradient: the interplay of leaf temperature, heat tolerance, and transpiration driven by the energy balance
_______
This repository contains a Python-based leaf energy balance simulator developed to quantify biophysical interactions between plant traits and and the environment. With this model the leaf temperature and stomatal conductance can be calculated, and therefore the energy balance equation can besolved under varying climatic conditions.
## Biophysical foundations of the leaf energy balance 
The energy balance of a plant leaf is described as the equilibrium between energy uptake and release. The primary focus lies on the radiation balance, since energy is absorbed, converted, and re-emitted by the plant in the form of radiation. As a result, the net radiation can be calculated, which is the driving force behind additional energy fluxes. The plant employs various physiological and morphological mechanisms to maintain this balance while also being influenced by diverse external factors.
*The leaf energy balance equations are further described in detail in chapter 3 Leaf energy balance in the masterthesis.*
## Methodology and Implementation
In general, the energy balance of plants is described by various scientific references using different formulae with varying degrees of accuracy. Therefore the model derived from established biophysical frameworks: 
  - Experimental Plant Ecology by Dieter J. von Willert
  - Gates Biophysical Ecology (Gates, 1980; J. von Willert et al., 1995)
  - Reference model: Refinded basd on the Excel-Sheet framework provided by Dr. Markus Burghardt
------
The model can iteratively solves for the **leaf temperature**, where the sum of all energy fluxes equals zero.
It can also solve for the critical **stomatal conductance **, which allows to determine how close a plant is operation to its thermal tipping point (heat avoidance vs. heat tolerance).

Different leaf energy balance scenarios based on the model are discussed in Chapter X.
