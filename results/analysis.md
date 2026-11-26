# Analysis

## E.1 Why 75 % masking in the MAE and 15 % in BERT

Language and images differ in how much information a single unit carries. A word is a dense, semantic token: removing 15 % of the words in a sentence already leaves a difficult prediction problem, because a missing word can rarely be inferred from its spelling neighbours. Image patches are highly redundant. Neighbouring 8 x 8 patches of a satellite tile share colour, texture and edges, so with only 15 % of the patches removed a missing patch can be filled in by local interpolation, and the network would learn little beyond smoothing. Masking 75 % (48 of 64 patches) removes this shortcut: from 16 scattered patches the model has to infer the global structure of the scene, i.e. where a field boundary continues, which way a river or road runs and where a water body ends. A side effect is efficiency, since the encoder only processes a quarter of the tokens.

The reconstructions (`mae_reconstructions.png`) show exactly this. The final masked-patch MSE is 0.0067 on the training set and 0.0069 on the validation set. The predicted patches are blurry, because the MSE objective is minimised by the conditional mean of all plausible completions, but they reproduce the layout: the colour of large regions, the position of boundaries between fields, the course of roads and rivers and the extent of water. Fine, high-frequency texture is lost. What the model had to learn is therefore a representation of the spatial layout and land-cover composition of the whole tile, which is also the information the few-label classifier needs.

## E.2 Most frequent confusions of the few-label classifier

With only 160 labelled images, the linear classifier on frozen MAE features reaches 46.6 % test accuracy against a chance level of 10 %. The most frequently confused pairs (counts in both directions on the 5,400 test images) are:

- **AnnualCrop and PermanentCrop** (248 errors; AnnualCrop predicted as PermanentCrop: 197, PermanentCrop predicted as AnnualCrop: 51): both are cultivated farmland with regular parcel geometry; at 10 m resolution an orchard or vineyard looks like a field of annual crops.
- **HerbaceousVegetation and PermanentCrop** (216 errors; HerbaceousVegetation predicted as PermanentCrop: 89, PermanentCrop predicted as HerbaceousVegetation: 127): permanent crops in a sparse or early growth stage show the same mottled green texture as natural herbaceous cover.
- **Pasture and River** (170 errors; Pasture predicted as River: 67, River predicted as Pasture: 103): the two classes share similar colour and texture statistics at 64 x 64 pixels, which the averaged encoder feature does not separate with only 16 examples per class.

The common cause is that the MAE learns appearance, i.e. colour, texture and layout, without labels. Classes that are defined by land use rather than by appearance (farming practice, grazing, building purpose) occupy overlapping regions of this feature space, and 16 examples per class are not enough for a linear boundary to separate them.

## E.3 Reconstruction and KL terms of the VAE

Over 30 epochs the training reconstruction term decreased from 201.5 to 104.7 nats per image, and the KL term increased from 14.9 to 25.3 nats (maximum 25.3 at epoch 30). On the test set the final values are 104.8 and 25.3 nats. Early in training the decoder learns the average character shape and the reconstruction term drops quickly; as the encoder starts to place different characters at different positions in latent space, it pays for that information with a larger KL term. The two terms then settle into a trade-off in which every additional nat of KL must buy a larger reduction in reconstruction error. The KL term measures how much information about the image the 16-dimensional code carries.

If the KL term were removed, the objective would reduce to that of a (noisy) deterministic autoencoder. Nothing would tie the aggregate posterior to N(0, I): the encoder could shrink the variances towards zero and place the codes anywhere, at any scale, with large empty regions between characters. Reconstructions would become sharper, but samples drawn from N(0, I) would mostly land in regions the decoder never saw during training and would decode to blobs, broken strokes or mixtures of characters rather than recognisable letters. Interpolations would pass through the same unused regions. The KL term is what makes the prior a valid sampling distribution.

## E.4 VAE samples versus diffusion samples

The DDPM samples (`ddpm_samples_40x8.png`) are clearly sharper than the VAE samples (`vae_samples.png`): strokes have clean edges, consistent thickness and distinct dots, while the VAE samples are recognisable but soft and grey at the edges. The reasons follow from the two objectives. The VAE decodes an image in a single pass from a 16-dimensional code through a factorised per-pixel likelihood. When the code does not determine every detail, the likelihood is maximised by predicting the average of all plausible images, which is blurred, and the KL term further limits how much detail the code can carry. The DDPM is trained with L_simple to predict the noise at every noise level, which corresponds to a (reweighted) bound over 1,000 small Gaussian denoising steps. Each step only has to model a small, nearly Gaussian change, there is no low-dimensional bottleneck, and the fine detail is produced in the last low-noise steps, where the model is trained to restore exactly that detail. The price is speed: the VAE generates a character in one forward pass, the DDPM needs 1,000 network evaluations (0.400 s per image in this run, batched).

## E.5 Characters the diffusion model writes worst

*Pending: the blind test in `results/human_eval.csv` has not been filled in yet. Re-run Sections 8 and 10 afterwards.*

Urdu letters form families that share one base shape and differ only in the number and position of dots (for example be, pe, te, Te and se; jeem, che, he and khe; dal, Dal and zal; re, Re, ze and zhe; seen and sheen). At 28 x 28 pixels a dot covers only a few pixels, so a sample with a well-formed base stroke but a missing, extra or misplaced dot is read as another member of the same family. The errors above should be read with this in mind: a model that writes the base shape correctly can still be marked wrong because of a single dot.

## E.6 Questions on which retrieval failed

For 98 of 707 test questions (13.9 %) no relevant abstract appears among the top 10. Three of them, with the relevant and the retrieved abstracts, are listed in `results/retrieval_failures.md`:

- Q4213: *Is capmatinib effective for glioblastoma?* - the generator still answered, citing passages [3].
- Q4456: *What is F105-P?* - the generator said it did not know.
- Q3807: *Which main viral protein is targeted by the drug remdesivir?* - the generator still answered, citing passages [1].

The failures share a few causes. MiniLM was trained on general-domain sentence pairs, so rare biomedical terms, gene and drug names and abbreviations are split into many sub-word pieces whose embeddings carry little meaning; a question built around such a term is matched to abstracts that share its general topic rather than the specific entity. Many questions are short and generic ("What is the function of ...?") while the relevant abstract states the answer without repeating the question's wording, so a lexical or topical match to another abstract scores higher. Biomedical text is also token-dense: 18.3 % of the 150-word chunks exceed 256 tokens and are truncated, so the end of such chunks is never seen by the retriever. Finally, mean pooling over a whole chunk dilutes the one sentence that answers the question with the surrounding methods and results.

When retrieval fails, none of the five passages comes from a relevant abstract, so any citation is by definition imprecise. Where the generator still answered, the answer either restates related content from the retrieved passages, which may be supported by them but does not answer the question, or draws on the model's own knowledge, which is not supported by the passages. Whether each answer was supported is judged in the human evaluation (pending).

## E.7 Why Recall@k is lower than Hit@k

Hit@k counts a question as solved as soon as one relevant abstract is among the top k. Recall@k asks for the share of the relevant abstracts that were found, out of min(k, |R_q|). For a question with a single relevant abstract the two are identical, but BioASQ questions have on average 8.66 relevant abstracts, and finding one of them is much easier than finding all of them: the most similar abstract is often found, while others that answer the question in different words are ranked lower. Since Recall@k is greater than zero only when there is a hit and is at most one, Recall@k <= Hit@k always holds for every question and therefore on average. In this run, Hit@10 = 0.861 and Recall@10 = 0.602; at k = 1 both are equal (0.622 and 0.622) because the denominator min(1, |R_q|) is 1.
