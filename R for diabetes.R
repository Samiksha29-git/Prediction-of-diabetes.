#Read the csv file of the data that you are gonna use.
Health<-read.csv(file.choose(),header = T)
#Let's see what the data looks like.
head(Health)
Health_model<-glm(Diabetes_binary~HighBP+HighChol+CholCheck+BMI+Smoker+Stroke+
                    HeartDiseaseorAttack+PhysActivity+Fruits+Veggies+HvyAlcoholConsump+
                    AnyHealthcare+NoDocbcCost+GenHlth+MentHlth+PhysHlth+DiffWalk+Sex+Age+
                    Education+Income,family = binomial,data = Health)
summary(Health_model)
#Now we see which of the independent variables are statistically significant and which are not.
#Here all the variables which have *** are highly significant.
New_Health_model<-glm(Diabetes_binary~HighBP+HighChol+CholCheck+BMI+Stroke+
                        HeartDiseaseorAttack+HvyAlcoholConsump+
                        GenHlth+MentHlth+PhysHlth+DiffWalk+Sex+Age+
                        Income,family = binomial,data = Health)
summary(New_Health_model)
#We create a new model with only statistically significant variables to get more accuracy.
null<-glm(Diabetes_binary~1, family=binomial,data=Health)
anova(null,New_Health_model, test="Chisq")
Health$predprob<-round(fitted(New_Health_model),4)
head(Health)
#Now we find predicted probabilities of the variables.
#Thus will help us predict which individuals are at a risk of getting diabetes.
install.packages("ROCR")
library(ROCR)
Health$predprob<-fitted(New_Health_model)
pred<-prediction(Health$predprob,Health$Diabetes_binary)
perf<-performance(pred,"tpr","fpr")
plot(perf)
#We have plotted an ROC curve which will help us understand if our model is accurate.
#Here the curve is towards the true positive rate that means our model is accurate 
abline(0,1)

#Now let us develop a generalized equation to predict if the individual will get the disease.
#Diabetes_binary probability =-6.825183+0.759567*HighBP+ 0.572801*HighChol + 1.281050*BMI +
# 0.183734*Stroke + 0.265145*HeartDiseaseorAttack + -0.730512*HvyAlcoholConsump + 
# 0.594556*GenHlth + -0.005005*MentHlth + -0.006757*PhysHlth + 0.102786*DiffWalk +
# 0.270340*Sex + 0.143215*Age + -0.066582*Income


