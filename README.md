![competition](https://github.com/sin1012/Kaggle_Jigsaw_Solo_Gold_Medal/blob/master/submissions%20and%20results/competiton.png)
# <center> Kaggle: Jigsaw Multilingual Toxic Comment Classficiation
https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification

## Timeline
June 15, 2020 - Entry deadline. You must accept the competition rules before this date in order to compete.  

June 15, 2020 - Team Merger deadline. This is the last day participants may join or merge teams.  

June 01, 2020 - I joined the competition

June 22, 2020 - Final submission deadline.  

## Summary
In this competition, I finished with 12th place out of 1621 teams and I was awarded a **solo gold medal**. I joined pretty late(21 days before the competition deadline) and I didn't expect to get any medals/rewards or whatsoever. I am grateful that I was lucky enough to finish well in this competition. I saw many great kernels in this competition and those kernels really helped me get started. I also learned how to use **TPU** for faster training; in fact, most of my training are done in COLAB PRO with TPU support. Most of my models are **XLM-ROBERTA Large**, which is the largest model possible for BERT's variants. To improve the results, I ultilized some common techniques like **Pseudo-Labelling**, **Label Smoothing** **Two-Staged Training** and **Test Time Augmentatin**. I was able to achieve `0.9490` without any post-processing. Later on, I was able to discover some tricks for post processing and I improved my public leaderboard score to `0.9500` and `0.9488` for private leaderboard. Overall, I learned a lot in this competition and it has further developed my understanding of NLP.
![final standing](https://github.com/sin1012/Kaggle_Jigsaw_Solo_Gold_Medal/blob/master/submissions%20and%20results/final_leaderboard.png)
https://www.kaggle.com/underwearfitting

## So what is this competition about?
In brief, this competition requires each competitor to develop models to predict the probability of toxic comments in different languages including **Turkish, Portuguese, Russian, French and Italian** while only training with **English** dataset. The corresponding challenge would be the lack of approaches to develop a robust local validation as the training set only contains **English** dataset. Unlike most competitor, I did not use extra validations for final modelling: I merely used the provided validation set for local validation and observed that XLM-ROBERTA doesn't converge until getting a high validation score; I was able to get a validation score of `0.9600+`, however the leaderboard does not reflect that and then I decided to train with the whole dataset with `3` to `5` epochs according to the public leaderboard feedbacks.

## Pseudo-Labelling
I used several base XLM-ROBERTA Large models(blending scored `0.9466`) and made inferences on the testing dataset as pseudo labels(https://www.kaggle.com/blobs/download/forum-message-attachment-files/746/pseudo_label_final.pdf). I selected confident(probability > 0.9 and probability < 0.1) as labels and added them to the original training set to re-train the models. With 7 XLM-ROBERTA large(different hyperparameters) with added pseudo labels, I blended and got a score of `0.9480`.

## Label Smoothing
I then applied label smoothing(https://papers.nips.cc/paper/8717-when-does-label-smoothing-help.pdf) to re-train all the models. I didn't apply any advanced label smoothing but basic label smoothing as suggested by [Alex Shonenkov](https://www.kaggle.com/shonenkov): 
```python
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)
```
Basically, the smoothed label is (1-alpha) * one_hot_encoded_label + alpha/k, where k is the number of classes(in this case, `toxic` and `non-toxic`). After this the public leaderboard score improved to `0.9481`. The improvement was not significant, however, later on I observed a more significant boost in private leaderboard.

## Two-staged Training
Before this competition, I did not know much about **multi-stage training** until I saw this awesome [kernel](https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta) by [xhlulu](https://www.kaggle.com/xhlulu). He showed me how to fine-tune the model on validation dataset. Adding this to the training, my models significantly improved(`0.9485`).

## Test Time Augmentation
I used the [translated data](https://www.kaggle.com/kashnitsky/jigsaw-multilingual-toxic-test-translated) to make inferences on all **6 languages**(5 other languages + english) and blended the predictions with the original predictions. With this method, I was able to achieve `0.9490` on public leaderboard. At this point, many teams are doing the final blendings and some surpassed me. To maintain my position, I discovered some naive approaches for post-processing and they worked.

## Post Processing
Since the competition metric is sensitive to the global ranking of all predictions but not to a specific language, I decided to adjust predictions based on different languages. This is purely from observations from historical submissions and some guesses with luck. Then I blended some public kernels submissions, I was able to finish at `0.9500`.
```python 
test.loc[test["lang"] == "es", "toxic"] *= 1.05
test.loc[test["lang"] == "fr", "toxic"] *= 1.05
test.loc[test["lang"] == "it", "toxic"] *= 0.99
test.loc[test["lang"] == "pt", "toxic"] *= 0.99
test.loc[test["lang"] == "tr", "toxic"] *= 0.99
```
However, I failed to submit the best sub(`0.9489`) but it didn't affect the final standing.
![subs](https://github.com/sin1012/Kaggle_Jigsaw_Solo_Gold_Medal/blob/master/submissions%20and%20results/submissions.png)

## Fin
This is my first serious competition on Kaggle and I truly learned a lot as a ML practitioner. I hope to compete in more competitions in the future.
