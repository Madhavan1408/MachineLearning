import numpy as np
import math
class LinearRegression:
    def __init__(self,learning_rates,num_iterations):
        self.learning_rates=learning_rates 
        self.num_iterations=num_iterations 
        self.weights=0
        self.bias=0
        self.lambdaR=1
        
    def fit(self,x,y):
        n_samples=x.shape[0]
        
        
        for _ in range(self.num_iterations):
            y_predict=np.dot(x,self.weights) + self.bias 
            
            # Error calculation
            error=y-y_predict 
            
            # loss function calculation 
            cost_func=np.mean(error**2)
            
            # Gradient descent 
            dw=-(2/n_samples) * np.dot(error,x)
            db=-(2/n_samples) * np.sum(error)
            
            # update parameters 
            self.weights-=self.learning_rates*dw
            self.bias-=self.learning_rates*db 
    def predict(self,x):
        return self.weights * x+self.bias 
    def lasso_regression(self,x,y):
        n_samples=x.shape[0]
        for i in range(self.num_iterations):
            y_predict=self.weights * x +self.bias 
            
            # Error Calculatioon 
            error=y-y_predict 
            
            # Loss Function 
            cost_func=np.mean(error ** 2) + self.lambdaR * np.sum(np.abs(self.weights))
            
            # Gradient Descent 
            dw=-(2/n_samples) * np.dot(error,x) + (self.lambdaR * np.sign(self.weights))
            db=-(2/n_samples) * np.sum(error)
            
            # update Parameters
            self.weights-=self.learning_rates*dw 
            self.bias-=self.learning_rates*db
            
        
    def ridge_regression(self,x,y):
        n_samples=x.shape[0]
        self.lambdaR=1
        for i in range(self.num_iterations):
            y_predict=self.weights * x +self.bias 
            # Error calculation
            error=y-y_predict 
            
            # loss function 
            cost_func=np.mean(error**2) + (self.lambdaR *np.sum(self.weights **2))
            
            # Gradient Descent 
            dw=-(2/n_samples) * np.dot(error,x) + (2 * self.lambdaR * self.weights)
            db=-(2/n_samples) * np.sum(error)
            
            # update parameters
            self.weights-=self.learning_rates*dw
            self.bias-=self.learning_rates*db
            
    @staticmethod
    def mae(y_actual,y_predict):
        return np.mean(np.abs(y_actual-y_predict))
    
    @staticmethod
    def mse(y_actual,y_predict):
        return np.mean((y_actual-y_predict)**2)
    
    @staticmethod
    def rmse(y_actual,y_predict):
        return np.sqrt(np.mean((y_actual-y_predict)**2))
    
    @staticmethod
    def r_squared(y_actual,y_predict):
        ss_res=np.sum((y_actual-y_predict) **2)
        ss_total=np.sum((y_actual-np.mean(y_actual))**2)
        if ss_total==0:
            raise ValueError("R2 is undefined when actual values are not defined")
        return 1-(ss_res/ss_total)
            
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([2,4,6,8,10,12,14,16,18,20])

model=LinearRegression(learning_rates=0.01,num_iterations=1000) 
#model.ridge_regression(x,y)
model.lasso_regression(x,y)
z=10
actual=np.array([20])
predict=model.predict(np.array([z]))     
print(predict)     
print(LinearRegression.mae(actual,predict))