[8:53 pm, 30/5/2026] Sumit Kumar Mtech Guide Symbiosis Pune: sentence appears incomplete after “70”. Please specify the exact train/validation/test proportions, whether splitting was patient-, field-, farm- or image-level, how duplicate or near-duplicate images were handled, and whether class imbalance was controlled across all splits. Without this information, the reported accuracy and cross-validation stability cannot be reliably interpreted.

7. Provide stronger evidence for the explanation module. The manuscript reports IoU between Grad-CAM++ heatmaps and expert symptom annotations, but the annotation protocol, number of annotated images, inter-rater agreement, thresholding method and evaluation procedure are not sufficiently transparent. Please provide representative success and failure cases, compare Grad-CAM++, Grad-CAM, LayerCAM, Integrated Gradients, LIME and SHAP under identical settings, and explain why the chosen method is most suitable for agricultural stakeholders rather than only computationally convenient.

8. Clarify uncertainty estimation and calibration. Monte Carlo Dropout is potentially useful, but the paper should define how uncertainty is computed, how aleatoric and epistemic uncertainty are separated, how temperature scaling is performed, and how ECE, MCE, reliability diagrams and Brier score are calculated. Please also explain how uncertainty affects the final recommendation: does high uncertainty trigger expert referral, reduced treatment confidence, or alternative diagnoses?

9. Rework the LLM-based explanation and recommendation component. The manuscript must explain the prompt structure, input variables, safeguards against hallucinated treatment advice, source and validation of the agronomic knowledge base, and whether generated recommendations were reviewed by plant-pathology experts. Since inappropriate pesticide or treatment advice can have economic and environmental consequences, the system should include explicit safety boundaries and referral mechanisms.

10. Expand the human-subjects evaluation cautiously. The user study with 12 participants is interesting but too small for strong general claims. Please describe recruitment, consent, ethics approval or exemption, task randomisation, statistical testing, effect sizes, correction for multiple comparisons, and limitations. The revised manuscript should frame the study as preliminary unless a larger validation is added.

11. Improve experimental comparisons. The baseline section should include not only ResNet50, EfficientNet-B0, Inception-v3 and traditional machine-learning methods, but also recent fine-grained classification and lightweight vision models where appropriate. Ablations should isolate the effect of backbone choice, feature pyramid design, explanation loss terms, uncertainty estimation, language explanation and recommendation guidance. Each component should be justified by measurable benefit.

12. Enhance the visual-computing relevance for The Visual Computer. The current agricultural application is valuable, but the connection to visual computing should be made sharper. Please emphasise visual explanation design, visual evidence localisation, uncertainty-aware visual interfaces, human-centred interpretation of image models, and deployable visual decision-support systems. The figures should become analytical visualisations rather than decorative system diagrams.
13.
[8:53 pm, 30/5/2026] Sumit Kumar Mtech Guide Symbiosis Pune: 13. Correct presentation and language problems thoroughly. The manuscript contains many grammatical errors, typographical issues, awkward sentences and unresolved references such as “Table ??” These problems currently obscure the technical content. The revised manuscript should undergo professional language editing and should ensure that all tables, equations, figures and references are complete, consistently numbered and referred to correctly.

14. Make the work reproducible. For a non-review article, I strongly recommend open-sourcing the code on GitHub or Zenodo with a permanent link, complete documentation, environment files, trained-model instructions, DOI, and a citation note related to this submission to The Visual Computer. If the dataset itself cannot be redistributed, please provide scripts, split files, preprocessing details and evaluation protocols sufficient for replication.

15. Moderate deployment claims. The manuscript reports mobile feasibility, storage requirements, inference time and battery consumption estimates. Please clarify which results are measured and which are estimated, list device specifications, testing protocols, runtime settings and quantisation details, and discuss trade-offs between offline LLM explanations and practical field deployment.

16. Recommended recent references for consideration

1. A visual-language foundation model for disease diagnosis and doctor–patient co-decision. The Visual Computer, 2025: 1-13.
2. Shapley value-based class activation mapping for improved explainability in neural networks. The Visual Computer, 2025: 1-19.
3. A new human-computer interaction paradigm: Agent interaction model based on large models and its prospects. Virtual Reality & Intelligent Hardware, 2025, 7(3): 237-266.
4. Visual-language foundation modelsandartificial intelligence agents in healthcare: Bridging from technological innovation to clinical impact, Computational Visual Media, 2026
5. Talk With Socrates: Relation Between Perceived Agent Personality and User Personality in LLM-Based Natural Language Dialogue Using Virtual Reality. Computer Animation and Virtual Worlds, 2025, 36(3): e70033.

These articles are merely recommendations based on the content and research direction of the papers, which can help enhance the relevance of the paper topics and the timeliness of the content. However, they are not mandatory. Authors can choose whether to adopt them or not based on their own circumstances.

5. Preprint submission

To help your research reach a broader audience quickly and enhance its academic visibility, consider submitting a preprint of your manuscript to the In Review platform of Springer Nature journals (submission link: [https://www.researchsquare.com/publishers/in-review](https://www.researchsquare.com/publishers/in-review)) after your manuscript has been submitted to our journal. If you choose to do so, access the platform through the provided link, click the “Submit a Preprint” button in the upper right corner to enter the submission page, and then follow the on-screen instructions to register an account and upload your manuscript and related information.

6. Required response from the authors

Please explicitly answer the following three questions one by one in your response letter, and provide corresponding revisions and explanations in the revised manuscript accordingly:
(1) Does the manuscript clearly articulate a reusable problem formulation, benchmark, dataset, codebase, taxonomy, or methodological insight that other researchers can cite and build upon?
(2) Are the title, abstract, keywords, contribution statement, figures, and experimental comparisons sufficiently discoverable by researchers outside the immediate subfield?
(3) Are code, data, pretrained models, demos, or benchmark protocols made available, when ethically and technically feasible?

In summary, I see potential in this work because it addresses a practically important problem and attempts to move beyond prediction accuracy towards interpretable, uncertainty-aware and user-centred agricultural AI. Nevertheless, the current version must be reshaped from a promising prototype description into a rigorous visual-computing research article. The revision should make the scientific contribution reusable, the experimental evidence verifiable, the user-study claims proportionate, the deployment analysis measurable, and the writing substantially clearer. If these issues are addressed carefully, the manuscript may become a more compelling contribution to explainable visual decision support and applied visual intelligence.

The Visual Computer, as an important journal in computer graphics and visual computing, looks forward to receiving relevant, interesting and important contributions from your future research. When you submit your future research, please pay attention to comparing and analyzing your work in relation to relevant studies recent published in our journal, so as to enhance the relevance of your manuscript to the journal’s scope.

TVC editor.

When your revision is ready, the original submitting author, Sumit Kumar, must upload a revised manuscript and a point-by-point response. To view any reviewer reports, editor feedback, and the instructions for submitting your revision, please visit your submission details page: