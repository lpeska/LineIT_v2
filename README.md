# LineIT_v2
This repository contains the source codes for the LineIT tool: Lineup assemblIng Tool. The tool was presented at TIR 2019 workshop, for more details, see https://link.springer.com/chapter/10.1007%2F978-3-030-27684-3_25

## Abstract:
Photo lineups are an important method in the identification process for the prosecution and possible conviction of suspects. Incorrect lineup assemblies have led to the false identification and conviction of innocent persons. One of the significant errors which can occur is the lack of lineup fairness, i.e., that the suspect significantly differs from other candidates. 
Despite the importance of the task, few tools are available to assist police technicians in creating lineups. Furthermore, these tools mostly focus on fair lineup administration and provide only a little support in the candidate selection process. In our work, we first summarize key personalization and information retrieval (IR) challenges of such systems and propose an IR/Personalization model that addresses them. Afterwards, we describe a LineIT tool that instantiate this model and aims to support police technicians in assembling unbiased lineups. 

Currently, the LineIT tool is in the stage of proof-of-concept to evaluate selected information retrieval model. Some of the features of the final software (security, authentization, permission policy, lineup export, dynamic datasets etc.) are not supported yet.

## Main features:
- Query by (multiple) examples
- Local-focus of the query on particular detail of the face (ears, forehead etc.)
- Conjunctive filtering for content-based attributes
- Recommending candidates based on partially constructed lineup (attribute-based and visual-based similarity)

## Working demo:
http://herkules.ms.mff.cuni.cz/lineit/index.py
