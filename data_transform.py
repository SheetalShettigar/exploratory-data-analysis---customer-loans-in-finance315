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
       #Describe the data and check the data for null columns, categorical columns 
       # and total number of rows and columns

        print(self.df.info())
        print(self.df.describe())
        print(self.df['grade'].value_counts())
        print(self.df['sub_grade'].value_counts())
        print(self.df['home_ownership'].value_counts())
        print(self.df['verification_status'].value_counts())
        print(self.df['loan_status'].value_counts())
        print(self.df['employment_length'].value_counts())
        print(self.df.shape)
        print('percentage of null values in each column:')
        print(self.df.isnull().sum()/len(self.df))    
                
                
    
    def transform_data(self):

        # Columns dropped as these are identification fields not useful
        #self.df.drop(columns=['id','member_id'],inplace = True)

        # Columns dropped as more than 50% data had null values
        self.df.drop(columns=['mths_since_last_major_derog','mths_since_last_record','mths_since_last_delinq','next_payment_date'],inplace = True)
        
        #drop rows where term is null
        self.df = self.df.dropna(subset=['term'])

        #drop rows where int_rate is null
        self.df = self.df.dropna(subset=['int_rate'])
        
        #drop rows where last_credit_pull_dates is null <1%
        self.df = self.df.dropna(subset=['last_credit_pull_date'])

        #drop rows where last_payment_date is null <1%
        self.df = self.df.dropna(subset=['last_payment_date'])

        # Impute null values in column = collections_12_mths_ex_med with median value
        #print(df['collections_12_mths_ex_med'].value_counts())
        Plotter.qq_plot(self,self.df,'collections_12_mths_ex_med')
        self.df['collections_12_mths_ex_med'] = self.df['collections_12_mths_ex_med'].fillna(self.df['collections_12_mths_ex_med'].median())

        # funded_amount replaced with loan_amount where funded_amount was null
        # as more than 70% of the data has funded_amount = loan_amount
        self.df['funded_amount'] = self.df['funded_amount'].fillna(self.df['loan_amount'])

        #drop rows where employment_length is null
        self.df = self.df.dropna(subset=['employment_length'])
             
        #self.df.to_csv("loan_payments_transformed.csv",sep=',',index=False,encoding='utf-8')

        print(self.df.isnull().sum()/len(self.df)) 

        for i in self.df.columns:

            if self.df[i].dtype == object and 'date' not in i:
                self.df[i] = self.df[i].astype('string')

            if 'date' in i :
                self.df[i] = pd.to_datetime(self.df[i])

        #remove string 'months' from the term and convert to numeric column                     
        self.df['term'] = self.df['term'].str.replace(' months','')
        self.df['term'] = pd.to_numeric(self.df['term'])
             

        #remove 'years' and signs from employment length column and convert to string

        emp_len_dict = {"employment_length": {"10+ years": '10', "9 years": '9',"8 years": '8',"7 years":'7',
                                               "6 years": '6',"5 years": '5', "4 years": '4', "3 years": '3',
                                                "2 years": '2', "1 year": '1', "< 1 year": '0'}}
        
        self.df = self.df.replace(emp_len_dict)
        self.df['employment_length'] = pd.to_numeric(self.df['employment_length'])

        #After transforming the data Check for nulls
        print('percentage of null values in each column:')
        print(self.df.isnull().sum()/len(self.df))       
        self.df.info()
         

       
    def check_skew(self):
        
        skew_transform_df = pd.DataFrame()  # empty dataframe for data after skew transformation
        threshold = 0.5
        numeric_columns = self.df.select_dtypes(include='number').columns
        for i in numeric_columns[3:]: 
            #Original plot
            original_skew =  self.df[i].skew() 
            label = 'Original'
            Plotter.histplot(self,self.df,i,original_skew,label)
            
            print("------------------------------------------")
            
            if abs(original_skew) > threshold and original_skew > 0:     
                print(f"this indicates that column {i} is a positive skew {original_skew}")
                #Make the values positive
                minimum = np.amin(self.df[i])
                #If minimum is negative, offset all values by a constant to move all values to positive teritory
                if minimum <= 0:
                    self.df[i] = self.df[i] + abs(minimum) + 0.01

                #Perform BOXCOX Transformation
                skew_transform_df[i] = stats.boxcox(self.df[i])[0]
                new_skew = skew_transform_df[i].skew()
                label = 'Transformed'
                Plotter.histplot(self,skew_transform_df,i,new_skew,label)
                
            elif abs(original_skew) > threshold and original_skew < 0:
                print(f"this indicates that column {i} is negative skew {original_skew}")
                #Make the values positive
                minimum = np.amin(self.df[i])
                #If minimum is negative, offset all values by a constant to move all values to positive teritory
                if minimum <= 0:
                    self.df[i] = self.df[i] + abs(minimum) + 0.01

                #Perform BOXCOX Transformation
               
                skew_transform_df[i] = stats.boxcox(self.df[i])[0]
                
                new_skew = skew_transform_df[i].skew()
                label = 'Transformed'
                Plotter.histplot(self,skew_transform_df,i,new_skew,label)               
                
            else:
                
                print(f"this indicates data in column {i} is not skewed and transformation is not required")
                        

        return skew_transform_df
    
    def check_outliers(self):        
        
        numeric_columns = self.df.select_dtypes(include='number').columns
        for cols in numeric_columns[3:]:
            
            #Scatter plot to visualise outliers
            label = 'Original'
            Plotter.scatterplot(self,self.df,label,cols)
            #sns.scatterplot(x=self.df['id'],y=self.df[cols])
            #plt.title(f'{self.df[cols]}: Original Scatter plot with outliers')
            #plt.show()  

            #Detecting outliers using InterQuartile range method
            Q1 = self.df[cols].quantile(0.25)
            Q3 = self.df[cols].quantile(0.75)

            # Calculate IQR
            IQR = Q3 - Q1

            # Identify outliers
            outliers_df = self.df[(self.df[cols] < (Q1 - 1.5 * IQR)) | (self.df[cols] > (Q3 + 1.5 * IQR))]
            if outliers_df.empty:
                print(f"{cols} ------------------- no outliers")
            else:
                #print(cols)   
                #print(outliers_df.shape)
                outliers_removed_df = self.df[~((self.df[cols] < (Q1 - 1.5 * IQR)) | (self.df[cols] > (Q3 + 1.5 * IQR)))]
                label = 'Transformed'
                Plotter.scatterplot(self,outliers_removed_df,label,cols)
                #sns.histplot(self,transformed_df[cols],label,cols)
                #plt.title(f'{self.df[cols]}: Scatter plot with outliers removed')
                #plt.show()  


        return outliers_removed_df

    
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

                
        
    
    def percentage_recovered(self):
        #df_percent = pd.DataFrame()
        total_amount_funded_count = 0
        funded_amount_inv_count = 0
        percentage_data = []
                
        for index, row in self.df.iterrows():
            if row['total_payment'] >= row['loan_amount'] and row['loan_status'] == 'Fully Paid':
                total_amount_funded_count+=1

            if row['total_payment_inv'] >= row['funded_amount_inv'] and row['loan_status'] == 'Fully Paid':
                funded_amount_inv_count+=1

            if row['out_prncp'] > 0 and (row['loan_status'] == 'Current' or row['loan_status'] == 'In Grace Period')  :
                amount_recovery_nxt_six_mths = row['instalment'] * 6
                percentage_recovery_in_six_mths = (amount_recovery_nxt_six_mths * 100)/row['funded_amount']

                percentage_data.append(percentage_recovery_in_six_mths)
                

        total_amt_funded_percent = (total_amount_funded_count * 100)/len(self.df)
        funded_amt_inv_percent = (funded_amount_inv_count * 100)/len(self.df)
        
        data = {'Percentage': [round(funded_amt_inv_percent,2),round(total_amt_funded_percent,2)],
                'Name':['Inv funded amount recovered','total funded amount recovered']}
        
        df = pd.DataFrame(data)

        plt.pie(df['Percentage'],labels=df['Name'],autopct='%1.1f%%')
        plt.show()
        
        dict = {'Percentage Recovery in next 6 months': percentage_data}

        df_percentage = pd.DataFrame(dict)

        sns.histplot(df_percentage,bins=20)
        plt.show()

    def loss_to_company(self):
        total_count_charged_off = 0
        total_amt_recovered_bef_chargedoff = 0
        total_amt_funded = 0
        months_paid = 0
        remaining_months = 0
        revenue_loss = 0
        total_revenue_loss = 0
        no_of_mths_paid = 0
        revenue_loss_list = []
        term_left_list = []
        loss_dict = {}
        loss_df = pd.DataFrame()
        total_late_loan_count = 0
        total_late_loan_amount = 0    
        total_late_loan_paid_to_date = 0 
        late_loan_remaining_months = 0   
        total_late_loan_revenue_loss = 0
        total_revenue_expected = 0
        total_default_loan_revenue_loss = 0

        for index, row in self.df.iterrows():

            total_revenue_expected += row['instalment'] * row['term']

            if row['loan_status'] == "Charged Off": 
                total_count_charged_off += 1
                total_amt_recovered_bef_chargedoff += row['total_payment']
                total_amt_funded += row['loan_amount']
                #Calculate Revenue Loss
                months_paid = row['last_payment_date'] - row['issue_date']
                no_of_mths_paid =  int((int(str(months_paid).strip(" days 00:00:00"))/30))
                remaining_months = row['term'] - no_of_mths_paid
                revenue_loss = row['instalment'] * remaining_months
                total_revenue_loss += revenue_loss
                term_left_list.append(remaining_months)
                revenue_loss_list.append(revenue_loss)    

            if 'Late' in row['loan_status']:
                total_late_loan_count +=1
                total_late_loan_amount += row['loan_amount']    
                total_late_loan_paid_to_date += row['total_payment']
                late_loan_no_of_mths_paid =  int((int(str(row['last_payment_date'] - row['issue_date']).strip(" days 00:00:00"))/30))
                late_loan_remaining_months = row['term'] - late_loan_no_of_mths_paid
                late_loan_revenue_loss = row['instalment'] * late_loan_remaining_months
                total_late_loan_revenue_loss += late_loan_revenue_loss 

            if row['loan_status'] == 'Default':
                default_loan_no_of_mths_paid =  int((int(str(row['last_payment_date'] - row['issue_date']).strip(" days 00:00:00"))/30))
                default_loan_remaining_months = row['term'] - default_loan_no_of_mths_paid
                default_loan_revenue_loss = row['instalment'] * default_loan_remaining_months
                total_default_loan_revenue_loss += default_loan_revenue_loss
                
                
        loss_dict = {'Term_Left':term_left_list,'Revenue_Loss':revenue_loss_list}
        loss_df = pd.DataFrame(loss_dict)
                
        print(f'The percentage of Charged off loans: {round((total_count_charged_off * 100)/len(self.df),2)}')
        print(f'The total amount paid before charged off : {round(total_amt_recovered_bef_chargedoff,2)}')
        print(f'The total amount funded : {round(total_amt_funded,2)}')
        print(f'The projected loss of the loans marked as Charged Off : {round((total_amt_funded - total_amt_recovered_bef_chargedoff),2)}')
        print(f'The loss in revenue these loans would have generated for the company if they had finished their term : {round(total_revenue_loss,2)}')
        print(f'Percentage of Late Loans: {round((total_late_loan_count * 100)/len(self.df),2)}')
        print(f'Total number of customers with Late Loans: {round(total_late_loan_count,2)}')
        print(f'Loss the company would incur if late loans were changed to Charged Off: {round((total_late_loan_amount - total_late_loan_paid_to_date),2)}')
        print(f'Late Loans Revenue loss: {round(total_late_loan_revenue_loss,2)}')
        print(f'Percentage of Late Loans: {round((total_late_loan_count * 100)/len(self.df),2)}')
        print(f'Percentage Possible loss:{round((((total_default_loan_revenue_loss + total_late_loan_revenue_loss)) * 100)/total_revenue_expected,2)}')
        
        #plt.bar(loss_df['Revenue_Loss'],loss_df['Term_Left'])
        #loss_df.plot(x="Term_Left", y="Revenue_Loss", kind="bar", figsize=(10, 10))
        #plt.show()

    def loss_indicators(self):
        df_charged_off = pd.DataFrame()
        df_late_loan = pd.DataFrame()

        #create subset dataframe with Charged Off Loans
        selected_loan_status_co = ['Charged Off']        
        df_charged_off = self.df[self.df['loan_status'].isin(selected_loan_status_co)]

        #create subset dataframe with Charged Off Loans
        selected_loan_status = ['Late (31-120 days)','Late (16-30 days)']        
        df_late_loan = self.df[self.df['loan_status'].isin(selected_loan_status)]

        #Compare Grades of Charged Off - Customers

        print(df_charged_off['grade'].value_counts())

        #Compare Grades of Late - Customers  
        print(df_late_loan['grade'].value_counts())

        print('------------------------------------------------------------')    
        #Compare Homeownership status of Charged Off - Customers  
        print(df_charged_off['home_ownership'].value_counts())

        #Compare Homeownership status of Late - Customers
        print(df_late_loan['home_ownership'].value_counts())

        print('------------------------------------------------------------') 

        #Compare Purpose of Charged Off - Customers
        print(df_charged_off['purpose'].value_counts())

        #Compare Purpose of Late - Customers
        
        print(df_late_loan['purpose'].value_counts())
        


        
                
class Plotter:
    
    def __init__(self,df):
        #self.df = pd.read_csv("loan_payments.csv")
        self.df = df 
    
    def qq_plot(self,df,cols):
        #df = pd.read_csv("loan_payments.csv")
        qq_plot = qqplot(self.df[cols] , scale=1 ,line='q')
        plt.show()

    def histplot(self,df,column,skew,label):
        t=sns.histplot(df[column])
        #t.legend()
        plt.title(label + ' :' + str(column) + ' skew= ' + str(skew))
        plt.show()

    def scatterplot(self,df,label,cols):
        if label == 'Original': 
            plt.title(label+ f': Scatter plot with outliers: ' + str(cols))
            #sns.histplot(df[cols])
            sns.scatterplot(x=df['id'],y=df[cols],c='green',s=30)        
        elif label == 'Transformed' : 
            plt.title(label+ f': Scatter plot without outliers: ' + str(cols))
            #sns.histplot(df[cols]) 
            sns.scatterplot(x=df['id'],y=df[cols],c='green',s=30) 

        plt.show()  



 
if __name__ == "__main__":
    EDA_DataTransform=DataTransform()
    EDA_DataTransform.check_data()
    
    EDA_DataTransform.transform_data()
    
    outliers_removed_df = EDA_DataTransform.check_outliers()

    skew_transformed_df = EDA_DataTransform.check_skew()
    
    skew_transformed_df.describe()
    skew_transformed_df.info()

    EDA_DataTransform.check_colinearity()

    
    EDA_DataTransform.percentage_recovered()
    EDA_DataTransform.loss_to_company()
    EDA_DataTransform.loss_indicators()

   

   


    


        

