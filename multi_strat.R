rm(list=ls())                                                                    #Clear workspace.

dfd <- read.csv('train.csv')                                               #Get dataset to work with.

dfd$Survived <- ifelse(dfd$Survived==1,0,1)                                  #Recode died variable.
names(dfd)[names(dfd)=="Survived"] <- "died"                                     #Rename died variable.
dfd$Sex <- as.factor(toupper(substr(dfd$Sex,1,1)))                               #Recode gender.
# dfd$rid <- 1:nrow(dfd)                                                           #Generate row id.
# dfd <- dfd[,unique(c("rid",names(dfd)))]                                         #Put the row id column at the front.
dfd$complete <- ifelse(dfd$PassengerId %in% na.omit(dfd)$PassengerId,1,0)       #Build a variable to indicate complete cases.

set.seed(621)
dfd$tt <- with(dfd,
               ifelse(PassengerId %in% sample(PassengerId, size=0.5*nrow(dfd)),"tst","trn")
              )                                                                  #Split sample into test and train groups.

# Regression.
mdl <- glm(  died ~ Sex + Age + Pclass + SibSp + Parch + Fare + Embarked
           , data=subset(dfd, tt=="trn"), family="binomial")                     #Build logistic regression model.
summary(mdl)                                                                     #Present a standard model summary.
smr <- data.frame( OR =exp(coef(mdl))
                  ,l95=exp(confint(mdl))[,1]
                  ,u95=exp(confint(mdl))[,2]
                  ,pvl=summary(mdl)$coef[,"Pr(>|z|)"]
                 )                                                               #Present a summary of the odds ratios.
dfd$prd.glm <- predict(mdl, newdata=dfd, type="response")                        #Get prediction.
rm(mdl,smr)                                                                      #Clean up.

# Recursive partitioning.
require(randomForest)                                                            #Random forest.
require(party)                                                                   #Statistically based recursive partitioning.
require(gbm)                                                                     #Generalized boosted machines.

mdl <- randomForest(  as.factor(died) ~ Sex + Age + Pclass + SibSp + Parch + Fare + Embarked
                    , data=na.omit(subset(dfd, tt=="trn")))                      #Build random forest model...works.
varImpPlot(mdl)                                                                  #Can produce some useful plots within the package.
dfd$prd.rf <- predict(mdl, newdata=dfd, type="prob")[,2]                         #Get prediction.
rm(mdl)                                                                          #Clean up.

mdl <- ctree(  as.factor(died) ~ Sex + Age + Pclass + SibSp + Parch + Fare + Embarked
             , data=subset(dfd, tt=="trn"))                                      #Build conditional tree model..
plot(mdl)                                                                        #Fun plot.
dfd$prd.ctr <- unlist(predict(mdl, newdata=dfd, type="prob"))[1:(nrow(dfd)*2)%%2==0] #Get preciction...kind of painful.
rm(mdl)                                                                          #Clean up.

mdl <- cforest(   as.factor(died) ~ Sex + Age + Pclass + SibSp + Parch + Fare + Embarked
               ,  data=subset(dfd, tt=="trn"))                                   #Build conditional forest model..
dfd$prd.cfst <- unlist(predict(mdl, newdata=dfd, type="prob"))[1:(nrow(dfd)*2)%%2==0] #Get prediction...kind of painful.
rm(mdl)                                                                          #Clean up.

mdl <- gbm(  died ~ Sex + Age + Pclass + SibSp + Parch + Fare + Embarked
           , data=subset(dfd, tt=="trn"), distribution="bernoulli", n.trees=10^4) #Build generalized boosted model (gradient boosted machine).
bst <- gbm.perf(mdl,method="OOB")                                                #Show plot of performance and store best
dfd$prd.gbm <- predict(mdl, newdata=dfd, bst, type="response")                   #Get prediction.
rm(mdl, bst)                                                                     #Clean up.

# Neural net.
require(nnet)

mdl <- nnet(  as.factor(died) ~ Sex + Age + Pclass + SibSp + Parch + Fare + Embarked
            , data=na.omit(subset(dfd, tt=="trn")), size=2)                      #Build neural net.
mdl
dfd$prd.ann <- predict(mdl, newdata=dfd)                                         #Get prediction.
rm(mdl)                                                                          #Clean up.

# Support vector machines.
require(e1071)

mdl <- svm(  as.factor(died) ~ Sex + Age + Pclass + SibSp + Parch + Fare + Embarked
           , data=subset(dfd, tt=="trn"), probability=TRUE)                      #Build support vector machine.
dfd$prd.svm[dfd$complete==1] <- attr(predict(mdl, newdata=dfd, probability=TRUE), "probabilities")[,2] #Populated prediction...carefully.
rm(mdl)                                                                          #Clean up.

# Naive Bayes.
mdl <- naiveBayes(  as.factor(died) ~ Sex + Age + Pclass + SibSp + Parch + Fare + Embarked
                  , data=subset(dfd, tt=="trn"))                                 #Build Naive Bayes model.
mdl
dfd$prd.nb <- predict(mdl, newdata=dfd[,names(mdl$tables)], type="raw")[,2]      #Get prediction...another funny one to avoid issues with non-represented variables appearing in the dataset fed to "predict".
rm(mdl)                                                                          #Clean up.

# Ensemble.

mdl <- ctree(  as.factor(died) ~ Sex + Age
             , data=subset(dfd, tt=="trn"))                                      #Build conditional tree model..
plot(mdl)                                                                        #Fun plot.
dfd$ctr.nod <- predict(mdl, newdata=dfd, type="node")                            #Get nodes for data.
rm(mdl)                                                                          #Clean up.
dfd$prd.ens <- NA                                                                #Allocate column for predictor.
nds <- sort(unique(dfd$ctr.nod))                                                 #Get unique list of nodes.
for(i in 1:length(nds)) {                                                        #Loop through nodes.
   mdl <- glm(  as.factor(died) ~ Pclass
              , data=subset(dfd, tt=="trn" & ctr.nod==nds[i]), family="binomial") #Build logistic regression model within each node group.
   dfd[dfd$ctr.nod==nds[i],]$prd.ens <- predict(mdl, newdata=dfd[dfd$ctr.nod==nds[i],], type="response") #Get node level predictions...only for rows in the particular node.
   rm(mdl)                                                                       #Clean up.
}
rm(nds,i)                                                                        #Clean up.

# Produce summary AUC table.
require(caTools)

colAUC(X=dfd$prd.glm, y=dfd$died)                                                #Keep in mind that there are some missing values that get ommitted.
smr <- data.frame( all=as.numeric(colAUC(X=dfd[,names(dfd)[grep("prd",names(dfd))]], y=dfd$died))
                  ,trn=as.numeric(colAUC(X=subset(dfd, tt=="trn")[,names(dfd)[grep("prd",names(dfd))]], y=subset(dfd, tt=="trn")$died))
                  ,tst=as.numeric(colAUC(X=subset(dfd, tt=="tst")[,names(dfd)[grep("prd",names(dfd))]], y=subset(dfd, tt=="tst")$died))
                 )                                                               #Produce a dataframe with all results and split into testing and training.
rownames(smr) <- names(dfd)[grep("prd",names(dfd))]                              #Change the rownames to make them more readable.
smr <- smr[order(smr$tst, decreasing=TRUE),]                                            #Print the results based on best testing performance.
save(smr, file = 'modelsummary.Rda')

