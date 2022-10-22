
import pandas as pd
import numpy as np
import re
from sklearn import linear_model as LM, preprocessing as pp, metrics, tree
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier as MLPC, MLPRegressor as MLPR

#Seed Numpy for Neural Nets
#I found the best success overall starting at a seed of 1000,
#Not sure if this is the best way to do it, but it worked for me
#but I have No Idea why it works
np.random.seed(1000)


class Model():
    def __init__(self, trn, tst):
        self.le=pp.LabelEncoder() #Label Encoder
        self.sc = StandardScaler() #Standard Scaler
        
        #Useless Columns once data is PreProcessed
        self.uselessColumns = ['Survived?', 'Class/Dept', 'URL', 'Body','Ticket', 'Cabin', 'Died', 'Boat','Born', 'Occupation', 'Name'] 
        
        self.ogtrain = trn #Original Train
        self.ogtest = tst #Original Test
        self.train = self.PreProcess(trn) #Preprocessed Train
        self.test = self.PreProcess(tst) #Preprocessed Test
        self.survive_Trn_X, self.survive_Trn_y = self.xySurvive(self.train) #Unpack Survival Train
        self.survive_Tst_X, self.survive_Tst_y = self.xySurvive(self.test) #Unpack Survival Test
        self.fare_Trn_X, self.fare_Trn_y = self.xyFare(self.train) #Unpack Fare Train
        self.fare_Tst_X, self.fare_Tst_y = self.xyFare(self.test) #Unpack Fare Test
        self.sc.fit(self.survive_Trn_X) #Survival Scaler only on Train
        self.scaled_Survive_Trn_X = self.sc.transform(self.survive_Trn_X) #Scaled Survival Train
        self.scaled_Survive_Tst_X = self.sc.transform(self.survive_Tst_X) #Scaled Survival Test
        self.sc.fit(self.fare_Trn_X) #Fare Scaler only on Train
        self.scaled_Fare_Trn_X = self.sc.transform(self.fare_Trn_X) #Scaled Fare Train with Survival Scaler based on Train ONLY
        self.scaled_Fare_Tst_X = self.sc.transform(self.fare_Tst_X) #Scaled Fare Test with Survival Scaler based on Train ONLY
        
    #map class feature to a number
    @staticmethod
    def dept(val) -> int:
        return int(val[0])

    #Convert Fare to Decimal
    #Pounds/Shillings/Pence to Pounds(decimal)
    @staticmethod
    def FixFares(val) -> int:
        val = str(val).split()
        pence = 0
        if val[0] == 'nan':
            return np.nan
        if len(val) == 1:
            return round(float(val[0][1:]),2)
        shillingsToPence = float(val[1][:-1])*12
        if len(val) > 2:
            pence = float(val[2][:-1])
        pounds = round(((shillingsToPence + pence)/240),2)
        pounds += int(val[0][1:])
        return pounds

    #map the survived feature to 1 or 0
    @staticmethod
    def Survived(val) -> int:
        if val == 'LOST':
            return 0
        elif val == 'SAVED':
            return 1
        else:
            return np.nan

    #ensure age feature is all integers
    @staticmethod
    def age(val) -> int:
        if str(val) == 'nan':
            return np.nan
        if type(val) == float:
            return int(val)
        else:
            return int(re.sub('[^0-9]','', val))
        
    #Preprocess the data
    def PreProcess(self, ogdf) -> pd.DataFrame:
        
        #Make a Deep copy of the dataframe, and reset/drop the index
        df = ogdf.copy(deep=True).reset_index(drop=True)
        
        #Apply Custom Survived Preprocessing
        df['Survived'] = df['Survived?'].apply(self.Survived)
        
        #Apply Custom Fare Preprocessing
        df['Fare'] = df['Fare'].apply(self.FixFares).astype(float)
        
        #Apply Custom Class Preprocessing
        df['Class'] = df['Class/Dept'].apply(self.dept)

        #Fill unknown(NaN) Fares to the mean
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
        #Apply Age Preprocessing
        df['Age'] = df['Age'].apply(self.age)
        
        #Fill unknown(NaN) Ages to the mean
        df['Age'] = df['Age'].fillna(df["Age"].median())
        
        #Use Label Encoder on Joined, Gender, and Nationality
        df['Joined'] = self.le.fit_transform(df['Joined'])
        df['Gender'] = self.le.fit_transform(df['Gender'])
        df['Nationality'] = self.le.fit_transform(df['Nationality'])
        
        #Drop Useless Columns
        df.drop(self.uselessColumns, axis=1, inplace=True)
        
        #Drop NaNs
        df.dropna(inplace=True)
        
        #Return the Cleaned DataFrame
        return df

    #Get the X and y for the survival model
    @staticmethod
    def xySurvive(df: pd.DataFrame) -> tuple:
        y = df['Survived']
        x = df.drop(['Survived'], axis=1)
        return (x, y)

    #Get the X and y for the fare model
    @staticmethod
    def xyFare(df: pd.DataFrame) -> tuple:
        y = df['Fare']
        x = df.drop(['Fare'], axis=1)
        return (x, y)

    #Print the survival score and Confusion Matrix
    def SurvivedScore(self, pred) -> None:
        print(f"Accuracy Score: {round(metrics.accuracy_score(self.survive_Tst_y, pred),2)*100}%")
        print(f"ConfusionMatrix:\n {metrics.confusion_matrix(self.survive_Tst_y, pred)}\n")

    #Print the fare score (R@, and MSE)
    def FareScore(self, pred) -> None:
        print(f"R2: {metrics.r2_score(self.fare_Tst_y, pred)}")
        print(f"MSE: {metrics.mean_squared_error(self.fare_Tst_y, pred)}\n")
    
    #Run the survival models
    def SurvivalPredictions(self) -> None:
        print("Survival Predictions:")
        print("___________________________________________\n")
        
        #Neural Net 1
        print("Multilayer Perceptron - solver: adam:")
        self.train_predict_survive(MLPC(solver='adam', max_iter=10000))
        print("___________________________________________\n")
        
        #Neural Net 2
        print("Multilayer Perceptron - solver: SGD:")
        self.train_predict_survive(MLPC(solver='sgd', max_iter=100000)) 
        print("___________________________________________\n")
        
        #Decision Tree
        print("Decision Tree Classifier:")
        self.train_predict_survive(tree.DecisionTreeClassifier())
        print("")
       
    #Run the fare models 
    def FarePredictions(self) -> None:
        print("\n____________________________________________________")
        print("____________________________________________________\n")
        
        print("\nFare Predictions:")
        print("___________________________________________\n")
        #Neural Net
        print("MultiLayer Perceptron Regressor - solver: adam, Activation: tanh")
        self.train_predict_fare(MLPR(max_iter=10000, activation='tanh', random_state=728, solver='adam', learning_rate='adaptive',))
        print("___________________________________________\n")
        #Linear Regression
        print("Linear Regression:")
        self.train_predict_fare(LM.LinearRegression())
        print("___________________________________________\n")
        #Decision Tree Regressor
        print("Decision Tree Regressor:")
        self.train_predict_fare(tree.DecisionTreeRegressor())
        print("___________________________________________\n")

    #Train and predict the survival model
    def train_predict_survive(self, classifier) -> None:
        #Fit the Model
        classifier.fit(self.scaled_Survive_Trn_X, self.survive_Trn_y)
        #Make Predictions
        pred = classifier.predict(self.scaled_Survive_Tst_X)
        #Print the Score
        self.SurvivedScore(pred)
        #Print the Classification Report
        print("ClassificationReport:")
        print(metrics.classification_report(self.survive_Tst_y, pred))       
    
    #Train and predict the fare model
    def train_predict_fare(self, classifier) -> None:
        #Fit the Model
        classifier.fit(self.scaled_Fare_Trn_X, self.fare_Trn_y)
        
        #Make Predictions
        pred = classifier.predict(self.scaled_Fare_Tst_X)
        
        #Print the Score
        self.FareScore( pred)
        
        

 
# Read in the data
ogTraindf = pd.read_csv('titanic_train.csv')
ogTestdf = pd.read_csv('titanic_test.csv')

#Drop Unnamed column
ogTraindf.drop('Unnamed: 0', axis=1, inplace=True)
ogTestdf.drop('Unnamed: 0', axis=1, inplace=True)

#Create the Model
model = Model(ogTraindf, ogTestdf)

#Run the survival models
model.SurvivalPredictions()
model.FarePredictions()



        