
#from Basic_Algorithms_FinalVersions.Classification_SVM_InPractice import use_linear_svm_to_train as svm_train
import statsmodels.api as sm

# Classification_SVM_CorrectVersion.main()

# Classification_SVM_InPractice.use_linear_svm_to_train()

# Neural_Network_Regression.make_data_weights_biases(4, twolayers=False)

data = sm.datasets.scotland.load()
data.exog = sm.add_constant(data.exog)

gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Binomial())
gamma_results = gamma_model.fit()
print(gamma_results.summary())










