import pandas as pd
from dateutil.parser import parse
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import yeojohnson


class DataTransform:

    def __init__(self):
        self.df = pd.read_csv("loan_payments.csv")
        
       

    def check_data(self):
        #df = pd.read_csv("loan_payments.csv")
        #print(df.info())
        for i in self.df.columns:
            #print(i)
            if self.df[i].dtype == object and 'date' not in i:
                self.df[i] = self.df[i].astype('string')

            if 'date' in i :
                #df[i] = df.i.apply(parse)
                self.df[i] = pd.to_datetime(self.df[i],errors='coerce')

                 
        self.df['term'] = self.df['term'].str.replace(' months','')
        self.df['term'] = pd.to_numeric(self.df['term'],errors='coerce')
             

        self.df['employment_length'] = self.df['employment_length'].str.replace(' years','')
        self.df['employment_length'] = self.df['employment_length'].str.replace(' year','')

        
                
    def describe_data(self):
        #df = pd.read_csv("loan_payments.csv")
        print(self.df.info())
        print(self.df.describe())
        print(self.df['grade'].value_counts())
        print(self.df['sub_grade'].value_counts())
        print(self.df['home_ownership'].value_counts())
        print(self.df['verification_status'].value_counts())
        print(self.df.shape)
        
        
    
    def check_nulls(self):
        #df = pd.read_csv("loan_payments.csv")
        print('percentage of null values in each column:')
        print(self.df.isnull().sum()/len(self.df))
        self.df.info()

        
    
    def transform_data(self):
        # Columns dropped as more than 50% data had null values
        self.df.drop(columns=['mths_since_last_major_derog','mths_since_last_record','mths_since_last_delinq','next_payment_date'],inplace = True)
        
        #drop rows where term is null
        #print(self.df['term'].value_counts())
        self.df = self.df.dropna(subset=['term'])

        #drop rows where term is null
        self.df = self.df.dropna(subset=['int_rate'])
        #print(self.df['int_rate'].median())
        #print(self.df['int_rate'].mean())
        #print(self.df['int_rate'].mode())

        #drop rows where last_credit_pull_dates is null <1%
        self.df = self.df.dropna(subset=['last_credit_pull_date'])

         #drop rows where last_payment_date is null <1%
        self.df = self.df.dropna(subset=['last_payment_date'])

        # Impute null values in column = collections_12_mths_ex_med with median value
        #print(df['collections_12_mths_ex_med'].value_counts())
        self.df['collections_12_mths_ex_med'] = self.df['collections_12_mths_ex_med'].fillna(self.df['collections_12_mths_ex_med'].median())

        # funded_ammount replaced with funded_amount_inv where funded_amount was null
        # as more than 70% of the data has funded_amount = funded_amount_inv
        self.df['funded_amount'] = self.df['funded_amount'].fillna(self.df['funded_amount_inv'])

        #drop rows where employment_length is null
        self.df = self.df.dropna(subset=['employment_length'])
             
        #df_comparison = df.loc[df['funded_amount'] != df['funded_amount_inv']]

        #print(df_comparison)

       

       
    def check_skew(self):
        #self.df = self.df.drop(columns=['id','member_id'])
        #transformed_df = self.df
        threshold = 0.5
        numeric_columns = self.df.select_dtypes(include='number').columns
        for i in numeric_columns[3:]: 
            #Original plot
            original_skew =  self.df[i].skew() 
            label = 'Original'
            Plotter.histplot(self,i,original_skew,label)
            #t=sns.histplot(self.df[i],label="Original Skew: %.2f"%(self.df[i].skew()) )
            #t.legend()
            #pyplot.show()
            print("------------------------------------------")
            
            if abs(original_skew) > threshold and original_skew > 0:     
                print(f"this indicates that column {i} is a positive skew {original_skew}")
                #Make the values positive
                minimum = np.amin(self.df[i])
                #If minimum is negative, offset all values by a constant to move all values to positive teritory
                if minimum <= 0:
                    self.df[i] = self.df[i] + abs(minimum) + 0.01

                #Perform BOXCOX Transformation
                self.df[i] = stats.boxcox(self.df[i])[0]
                new_skew = self.df[i].skew()
                label = 'Transformed'
                Plotter.histplot(self,i,new_skew,label)
                #t=sns.histplot(self.df[i],label="Transformed Skew: %.2f"%(self.df[i].skew()) )
                #t.legend()
                #pyplot.show()
            elif abs(original_skew) > threshold and original_skew < 0:
                print(f"this indicates that column {i} is negative skew {original_skew}")
                #Make the values positive
                minimum = np.amin(self.df[i])
                #If minimum is negative, offset all values by a constant to move all values to positive teritory
                if minimum <= 0:
                    self.df[i] = self.df[i] + abs(minimum) + 0.01

                #Perform BOXCOX Transformation
                #transformed_df[i] = self.df[i]
                self.df[i] = stats.boxcox(self.df[i])[0]
                
                new_skew = self.df[i].skew()
                label = 'Transformed'
                Plotter.histplot(i,new_skew,label)
                #print(f"new skew {self.df[i].skew()}")
                #t=sns.histplot(self.df[i],label="Transformed Skew: %.2f"%(self.df[i].skew()) )
                #t.legend()
                #pyplot.show()
                
            else:
                #transformed_df[i] = self.df[i]
                print(f"this indicates data in column {i} is not skewed and transformation is not required")
                        

        return self.df
    
    def check_outliers(self):        
        #self.df = self.df.drop(columns=['id','member_id'])
        numeric_columns = self.df.select_dtypes(include='number').columns
        for cols in numeric_columns[3:]:
            
            #Scatter plot to visualise outliers
            label = 'Original'
            Plotter.scatterplot(self,self.df['id'],self.df[cols],label,cols)
            #sns.scatterplot(x=self.df['id'],y=self.df[cols])
            #plt.title(f'{self.df[cols]}: Original Scatter plot with outliers')
            #plt.show()  

            #Detecting outliers using InterQuartile range method
            Q1 = self.df[cols].quantile(0.25)
            Q3 = self.df[cols].quantile(0.75)

            # Calculate IQR
            IQR = Q3 - Q1

            #print(f"Q1 (25th percentile): {Q1}")
            #print(f"Q3 (75th percentile): {Q3}")
            #print(f"IQR: {IQR}")

            # Identify outliers
            outliers_df = self.df[(self.df[cols] < (Q1 - 1.5 * IQR)) | (self.df[cols] > (Q3 + 1.5 * IQR))]
            if outliers_df.empty:
                print(f"{cols} ------------------- no outliers")
            else:
                #print(cols)   
                #print(outliers_df.shape)
                transformed_df = self.df[~((self.df[cols] < (Q1 - 1.5 * IQR)) | (self.df[cols] > (Q3 + 1.5 * IQR)))]
                label = 'Transformed'
                Plotter.scatterplot(self,transformed_df['id'],transformed_df[cols],label,cols)
                #sns.scatterplot(x=transformed_df['id'],y=transformed_df[cols])
                #plt.title(f'{self.df[cols]}: Scatter plot with outliers removed')
                #plt.show()  

        return transformed_df
    
    def check_colinearity(self):
        
        # Compute the correlation matrix
        numeric_df = self.df.select_dtypes(include='number')

        corr = numeric_df.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True

        # set thins up for plotting
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap
        sns.heatmap(corr, mask=mask,square=True, linewidths=.5, annot=False, cmap=cmap)

        variables = []
        for i in corr.columns:
            variables.append(i)

        # Adding labels to the matrix
        plt.xticks(range(len(corr)), variables, rotation=45, ha='right')
        plt.yticks(range(len(corr)), variables)

        #plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()

        #####check the threshold of correlated columns. if the threshold is above 0.9 remove the columns

        #threshold_funded_amount = self.df['funded_amount'].corr(self.df['funded_amount_inv'])
        if (self.df['funded_amount'].corr(self.df['funded_amount_inv'])) > 0.9:
            self.df = self.df.drop('funded_amount_inv',axis=1)

        #threshold_out_prncp = self.df['out_prncp'].corr(self.df['out_prncp_inv'])
        if (self.df['out_prncp'].corr(self.df['out_prncp_inv'])) > 0.9:
            self.df = self.df.drop('out_prncp_inv',axis=1)

        #threshold_total_payment = self.df['total_payment'].corr(self.df['total_payment_inv'])
        if (self.df['total_payment'].corr(self.df['total_payment_inv'])) > 0.9:
            self.df = self.df.drop('total_payment_inv',axis=1)

        return self.df
                
class Plotter:
    
    def __init__(self,df):
        #self.df = pd.read_csv("loan_payments.csv")
        self.df = df 
    
    def qq_plot(self):
        #df = pd.read_csv("loan_payments.csv")
        qq_plot = qqplot(self.df['collections_12_mths_ex_med'] , scale=1 ,line='q')
        plt.show()

    def histplot(self,column,skew,label):
        t=sns.histplot(self.df[column])
        #t.legend()
        plt.title(label + ' :' + str(column) + ' skew= ' + str(skew))
        plt.show()

    def scatterplot(self,x,y,label,cols):
        if label == 'Original': plt.title(label+ f': Scatter plot with outliers: ' + str(cols))
        elif label == 'Transformed' : plt.title(label+ f': Scatter plot without outliers: ' + str(cols))
        sns.scatterplot(x=x,y=y,c='green',s=30)        
        plt.show()  

 
if __name__ == "__main__":
    EDA_DataTransform=DataTransform()
    EDA_DataTransform.check_data()
    EDA_DataTransform.describe_data()
    EDA_DataTransform.check_nulls()

    EDA_DataTransform.transform_data()
    EDA_DataTransform.check_nulls()

    transformed_df = EDA_DataTransform.check_skew()
    
    transformed_df.describe()
   
    Outlier_Removed_Df = EDA_DataTransform.check_outliers()

    Outlier_Removed_Df.info()
    Outlier_Removed_Df.describe()

    Colinearity_Removed_Df = EDA_DataTransform.check_colinearity()

    Colinearity_Removed_Df.info()
    Colinearity_Removed_Df.describe()


    


        

