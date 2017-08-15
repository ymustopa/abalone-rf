# abalone-rf
The dataset (which has 9 features and 4,177 instances) and its attendant information can be obtained from the following URL:

https://archive.ics.uci.edu/ml/datasets/abalone/

We will carry out the regression task of predicting the age of blacklip abalone based on various physical measurements.  Although this was done with and without feature selection, what we present in detail is the approach via feature selection; for the other approach we present only the normalized RMS error for comparison.  Here are the descriptions of the variables.


|	Name |	Data Type	 | Units | Description |
|	---- | ---------	 | ----- | ----------- |
|	Sex	 | categorical |	N/A	 |	M (Male), F (Female), I (infant) |
|	Length | continuous |		mm	|	Longest shell measurement |
| 	Diameter | continuous |	mm | perpendicular to length |
|	Height	| continuous | mm	| with meat in shell |
|	Whole weight	|	continuous	|	grams	|    whole abalone |
|	Shucked weight | continuous | grams	|    weight of meat |
|	Viscera weight | continuous |	grams |	 gut weight (after bleeding) |
|	Shell weight	|	continuous	|	grams	 |   after being dried |
|	Rings		|	      integer	|		          |  +1.5 gives the age in years |

 ```````
 library(caret) 
 library(randomForest)
 library(ggvis)
  ```````
 Although there is an accompanying names file, we declare names which are unbroken and lowercase for relative ease of entry.
 ```````````
colnames(abalone)[1] <- “sex”        
colnames(abalone)[2] <- "length"
colnames(abalone)[3] <- "diameter"
colnames(abalone)[4] <- "height"
colnames(abalone)[5] <- "wholeweight"
colnames(abalone)[6] <- "shuckedweight"
colnames(abalone)[7] <- "visceraweight"
colnames(abalone)[8] <- "shellweight"
colnames(abalone)[9] <- "rings"
`````````````````
In order to prepare the dataset for training, we first reformat "sex" as a factor feature and normalize the numerical features, obtaining the dataframe "aba". 
``````````````````````
abalone$sex <- as.factor(unlist(abalone$sex))
aba <- as.data.frame(scale(abalone[,2:9]))
aba$class <- abalone[1]
aba <- aba[c(9,1,2,3,4,5,6,7,8)] 
``````````````````````
Before doing any visualizations, we inspect the correlation matrix for all numerical features (both input and response).
``````````````````````
print(cor(abalone[,2:9]))
`````````````````````````
|             |    length | diameter |   height |wholeweight |shuckedweight |visceraweight |shellweight  |    rings|
|-------------|-----------|----------|----------|------------|--------------|--------------|-------------|---------|
|length       | 1.0000000 |0.9868116 |0.8275536 |  0.9252612 |    0.8979137 |    0.9030177 |  0.8977056  |0.5567196|
|diameter     | 0.9868116 |1.0000000 |0.8336837 |  0.9254521 |    0.8931625 |    0.8997244 |  0.9053298  |0.5746599|
|height       | 0.8275536 |0.8336837 |1.0000000 |  0.8192208 |    0.7749723 |    0.7983193 |  0.8173380  |0.5574673|
|wholeweight  | 0.9252612 |0.9254521 |0.8192208 |  1.0000000 |    0.9694055 |    0.9663751 |  0.9553554  |0.5403897| 
|shuckedweight| 0.8979137 |0.8931625 |0.7749723 |  0.9694055 |    1.0000000 |    0.9319613 |  0.8826171  |0.4208837|
|visceraweight| 0.9030177 |0.8997244 |0.7983193 |  0.9663751 |    0.9319613 |    1.0000000 |  0.9076563  |0.5038192|
|shellweight  | 0.8977056 |0.9053298 |0.8173380 |  0.9553554 |    0.8826171 |    0.9076563 |  1.0000000  |0.6275740|
|rings        | 0.5567196 |0.5746599 |0.5574673 |  0.5403897 |    0.4208837 |    0.5038192 |  0.6275740  |1.0000000|

Given that quite a few of the numerical input features are very highly correlated, it makes sense to eliminate some of them.  We will do this using the findCorrelation function in caret with cutoff of 0.9, and obtain the dataframe "abareduced."  
````````
abacor <- cor(aba[,2:8])
aba_hc <- findCorrelation(abacor, cutoff=0.9)
aba_hc <- sort(aba_hc)
abareduced <- aba[,c(9,5,7,8)]
summary(abareduced)

 sex.sex     shuckedweight      shellweight          rings        
 F:1307      Min.   :-1.6145   Min.   :-1.7049   Min.   :-2.7708  
 I:1342      1st Qu.:-0.7811   1st Qu.:-0.7818   1st Qu.:-0.5997  
 M:1528      Median :-0.1053   Median :-0.0347   Median :-0.2896  
             Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000  
             3rd Qu.: 0.6426   3rd Qu.: 0.6478   3rd Qu.: 0.3307  
             Max.   : 5.0848   Max.   : 5.5040   Max.   : 5.9136   
````````
Below are ggvis plots of both "shuckedweight" and "shellweight" against "rings," with the "sex" feature taken into account via color-coding.  Afterwards, we split "abareduced" into training and testing sets.
``````
abareduced %>% ggvis(~shuckedweight, ~rings, fill=~as.factor(unlist(sex))) %>% layer_points()
abareduced %>% ggvis(~shellweight, ~rings, fill=~as.factor(unlist(sex))) %>% layer_points()
```````
![Shucked Weight vs. Rings](../master/shucked-plot.png) 

|[Shell Weight vs. Rings](../master/shell-plot.png)

```````
set.seed(1459)
split <- createDataPartition(y=abareduced$rings, p=0.75, list=FALSE)
trainingred <- abareduced[split,]
testingred <- abareduced[-split,]
```````
Before training random forest on "trainingred" we reformat the “sex” feature. 

````````
trainingred$sex <- as.factor(unlist(trainingred$sex))
testingred$sex <- as.factor(unlist(testingred$sex))
````````
We now set the control parameters and carry out the training.
```````````
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
rf_random <- train(rings~., data=trainingred, method="rf", metric="RMSE", tuneLength=15, trControl=control)
print(rf_random)

  Random Forest 

  3134 samples
  3 predictor

  No pre-processing
  Resampling: Cross-Validated (10 fold, repeated 3 times) 
  Summary of sample sizes: 2822, 2821, 2820, 2821, 2820, 2821, ... 
  Resampling results across tuning parameters:

  mtry  RMSE       Rsquared 
  1     0.7679243  0.4437757
  2     0.6865021  0.5342274
  3     0.7059635  0.5139921
  4     0.7134943  0.5062479

  RMSE was used to select the optimal model using  the smallest value.
  The final value used for the model was mtry = 2.
```````````  
To normalize the RMSE we use the range of the "rings" variable in "abareduced."
``````
> 0.6865021 / (max(abareduced$rings)-min(abareduced$rings))
[1] 0.07904996
``````
Finally we apply our model to the testing set.
```````
red_predict <- predict(rf_random, newdata=testingred)
red_rmse <- sqrt(mean((red_predict - testingred$rings)^2)) 
> print(rf_rmse)
[1] 0.6750398

> 0.6750398 / (max(abareduced$rings)-min(abareduced$rings))
[1] 0.07773009
```````
We conclude by comparing this result to random forest without feature selection.  All the previous code was applied to "aba" without any elimination of features.  Here are the resulting normalized RMSEs on the training and testing sets, respectively:
````````
[1] 0.07664232  [1] 0.07641347
````````
Although the implementation without feature selection took much longer, the difference in accuracy seems negligible.
