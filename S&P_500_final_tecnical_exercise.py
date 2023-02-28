# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:44:47 2022

@author: Leonardo
"""
import pandas as pd
import os
import numpy as np
import datetime as dt
from scipy.optimize import minimize
import warnings
from openpyxl import load_workbook

warnings.filterwarnings("ignore")


class InputsDescription:
    """ This class is used to define the files name and other parameters"""
    def __init__(self, inputs_path, inputs_file, inputs_sheet_name):
        self.inputs_path = inputs_path
        self.inputs_file = inputs_file
        self.inputs_sheet_name = inputs_sheet_name


class Calculations:
    """ This class is used to do the calculations necessaries for the 
    rebalance process """

    def __init__(self, data, method = 'SLSQP', min_wt = 0.5/1000, 
                 elements_index = 50): 
        self.data = data
        self.method = method
        self.min_wt = min_wt
        self.elements_index = elements_index
        
    def index_creation(self, data):
        """
        Function created to calculate the index
        @params:
            data: Initial global portfolio (DataFrame)
        @outputs:
            index: portfolio with 50 best firms in terms of Z_Value (DataFrame)
        """
        data.sort_values(by=data.columns[-1], ascending=False, inplace = True)
        index = data.iloc[0:50,:]
        return index
    
    def const_inputs(self, index: pd.DataFrame)-> pd.DataFrame:
        """ 
        Function created to calculate the inputs to be used for 
        the optimal calculations of weights
        @params:
            index: Index portfolio of the best 50 bonds
        @outputs:
            input_weigth: Index portfolio of the best 50 bonds adding some new columns:
                'FCap Wt*(1+Z_Value)', 'Uncapped Wt', 'Max Wt', 'Min Wt' (DataFrame)
        """
        input_weigth = index.copy() #Hago esto por que sin esto me copia el indice y eso no lo quiero 
        input_weigth['FCap Wt*(1+Z_Value)'] = input_weigth['FCap Wt']*input_weigth['Z_Value']
        input_weigth['Uncapped Wt'] = [item/input_weigth['FCap Wt*(1+Z_Value)'].sum() 
                                for item in input_weigth['FCap Wt*(1+Z_Value)'].tolist()]
        input_weigth['Max Wt_1'] = 20*input_weigth['FCap Wt']  
        input_weigth['Max Wt_2'] = 0.05
        input_weigth['Max Wt'] = input_weigth[['Max Wt_1', 'Max Wt_2']].min(axis=1)
        input_weigth['Min Wt'] = self.min_wt
        input_weigth.reset_index(inplace=True, drop=True)
        input_weigth.drop(['Max Wt_1', 'Max Wt_2'], axis=1, inplace=True)
        self.sectorWeight = input_weigth.groupby('Sector Code').agg({'Company Name': ['count'], 
                                                        'Uncapped Wt': ['sum']})
        self.sectorWeight.reset_index(inplace=True, drop=False)
        self.sectorWeight.columns = ['Sector Code', '# Stocks', 'Uncapped Wt']
        return input_weigth 
    
    def objectiveFunc(self, x: np.array, y: np.array) -> float:
        """ 
        Objective function to use in the optimization process
        @params:
            x: column 'Capped Wt'. As initial guess the value: 0.0005 is taken for each bond (np.array)
            y: column 'Uncapped Wt' (np.array)
        @outputs:
            count: sum of [(squared difference between Capped Wt and Uncapped Wt)/Uncapped Wt for each stock] (float)
        """
        count = 0
        for i in range(len(y)):
            count += (x[i]-y[i])**2/y[i]
        return count
    
    def constraint1(self, x: np.array) -> float: 
        """ 
        Equality constraint -> sum of all bonds weights equals to 1
        @params:
            x: column 'Capped Wt'. As initial guess the value: 0.0005 is taken for each bond (np.array)
        @outputs:
            result: diference of 1 and the total aggregation of each bond weight.(float)
        """
        result = sum(x)-1.0
        return result
    
    def constraint2(self, x: np.array, sector_list: list) -> float:
        """ 
        Inequality constraint -> sum of all bonds weights for each sector <= 0.5
        @params:
            x: column 'Capped Wt'. As initial guess the value: 0.0005 is taken for each bond (np.array)
        @outputs:
            result: diference of 0.5 and aggregation of weights for each sector. (float)
        """
        sum_sector = 0
        for i in sector_list:
            sum_sector = sum_sector + x[i]
        result = 0.5-sum_sector
        return result
    
    def constraints_creator(self, input_weigth: pd.DataFrame) -> list:      
        """ 
        Function which creates all the inequalities constrainst to be use in 
        the optimization process
        @params:
            input_weigth: Input with te information necessary for the construction 
            of the constraints (DataFrame)
        @outputs:
            cons: dictionaires in a list, each dictionary with the caracteristics 
            to be taken in cosideration in the optimization 
            process as:type of constrains, function, arguments. (list of dictionaries)
        """
        self.cons=[{'type':'eq','fun':self.constraint1}]
        sectors_list = []
        for i in input_weigth['Sector Code'].unique():
            sectors_list.append(input_weigth.index[input_weigth['Sector Code']==i].tolist())
        
        for i in range(len(sectors_list)):
            self.cons.append({'type':'ineq', 'fun': self.constraint2, 
                              'args': (sectors_list[i],)})
        return self.cons
    
    def bounds_creator(self, x:np.array, max_wt:np.array) -> list:
        """ 
        Function which stablish for each element, minimiun and maximun bounds
        @params:
            x:  values taken from the column 'MinValue' from the input (np.array)
            max_wt: values taken from the column 'Max Wt' from the input (np.array)
        @outputs:
            bounds: for each element to optimize its minimun and maximun bounds (list)
        """
        self.bounds = []
        for i in range(len(x)):
            self.bounds.append((x[i], max_wt[i])) #= input_june_17['MinValue'].values, input_june_17['Max Wt'].values
        return self.bounds
    
    def optimization_weigths(self, x: np.array, y: np.array) -> tuple:
        """ 
        Function which optimize the weights of the elements in the portfolio
        @params:
            x: column 'Capped Wt'. As initial guess the value: 0.0005 is taken for each bond (np.array)
            y: column 'Uncapped Wt' (np.array)
        @outputs:
            optimal_result: Outcome of the optimization process (tuple)
            (class: scipy.optimize.OptimizeResult)
        """
        optimal_result = minimize(self.objectiveFunc, x, args=(y), 
                                  method = self.method, bounds = self.bounds,
                                  constraints = self.cons)
        print(optimal_result.message)
        return optimal_result

    def index_opt_weigthed(self, index: pd.DataFrame, x: np.array) -> pd.DataFrame: 
        """ 
        Function which incoporate to the portfolio the optimized 'Capped Wt' values to the index
        @params:
            index: portfolio with 50 best firms ordered by Z_Value in descending order (DataFrame)
            x: Optimized 'Capped Wt' values (np.array)
        @outputs:
            index: portfolio with 50 best firms and its optimized 'Capped Wt' values (DataFrame)
        """
        index['Capped Wt'] = x
        return index
      
    def rebalance_process_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        ATENCION ACA--> la funcion no hace eso
        Function which rank all the stocks (current constituents or new candidates) ranked 
        within top 40 by Z_Value.
        @params:
            df: Stocks from the starting universe for a particular (DataFrame)

        @outputs:
            constituents_step1 : All the stocks (current constituents or new candidates) ranked 
            within top 40 by Z_Value are selected. (DataFrame)
        """               
        df.sort_values(by=df.columns[-1], ascending=False, inplace=True) 
        constituents_step1 = df.iloc[0:40,:]
        
        return constituents_step1
    
    def rebalance_process_2(self, index: pd.DataFrame, df: pd.DataFrame, 
                            constituents_step1: pd.DataFrame) -> pd.DataFrame:
        """        
        Second step in the rebalance process, this function add stocks to the current index. 
        Those stocks are ranked below top 40 but within top 60 and also appears in the former index.
        @params:
            index: Current index (DataFrame)
            df: Stocks from the starting universe for a particular (DataFrame)
            constituents_step1: The resulting index of the first step of the rebalance 
            process. (DataFrame)
        @outputs:
            constituents_step_1_2 : The resulting index of the first and second step of the 
            rebalance process. (DataFrame)
        """
        df.sort_values(by=df.columns[-1], ascending=False, inplace = True)
        const_step_2 = df.iloc[40:60,:]
        firms_not_step1_process = [item for item in index['Company Name'].tolist() 
                                  if item not in constituents_step1['Company Name'].tolist()]
        step2_constituents = pd.DataFrame(firms_not_step1_process, 
                                          columns= ['Company Name'])
        constituents_step2 = pd.merge(step2_constituents, const_step_2, 
                                      how="inner", on='Company Name',
                                      suffixes=("_x", "_y"))
        self.constituents_step2 = constituents_step2[['Ref Date', 'Company Name', 
                                                 'RIC','Sector Code', 'FCap Wt', 
                                                 'Z_Value']]
        constituents_step_1_2 = pd.concat([constituents_step1, self.constituents_step2])
        return constituents_step_1_2

    def rebalance_process_3(self, constituents_step_1_2: pd.DataFrame,
                            df: pd.DataFrame, index: pd.DataFrame, rest: int):
        """        
        Third step in the rebalance process, this function add stocks to the current index. 
         The stocks are selected in the order of their Z_Value scores untill the total number 
         of selected stocks reaches 50.
        @params:
            constituents_step_1_2 : The resulting index of the first and second step of the 
            rebalance process. (DataFrame)
            df: Stocks from the starting universe for a particular (DataFrame)
            index: Current index (DataFrame).
        @outputs:
            new_index : The resulting index of the first, second and third step of the 
            rebalance process. (DataFrame)
        """
        candidate = []
        for item in df['Company Name'].tolist():
            if item not in constituents_step_1_2['Company Name'].tolist():
                candidate.append(1)
            else:
                candidate.append(0)
        
        df['Candidate'] = candidate
        df.reset_index(inplace = True, drop=True)
        constituents_step_3 = df[df['Candidate']==1]
        constituents_step_3.sort_values(by='Z_Value', ascending=False, 
                                        inplace = True)
        self.constituents_step_3 = constituents_step_3.iloc[0:rest, 0:-1]   
        new_index = pd.concat([constituents_step_1_2, self.constituents_step_3]) 
        new_index.reset_index(inplace = True, drop=True)
        return new_index
    
    def main(self, index: pd.DataFrame, df_month_year: pd.DataFrame)  -> pd.DataFrame:
        """
        Function which contains all the rebalance process
        @params:
            index: Current index (DataFrame).
            df_month_date: Stocks from the starting universe for a particular (DataFrame)
    
        @outputs:
            final_index : The resulting index of the first, second and third step of the 
            rebalance process. (DataFrame)
        """
        elements_index = self.elements_index 
        constituents_step1 = self.rebalance_process_1(df_month_year)
        if len(constituents_step1['Company Name']) == elements_index:
            print('ok')
            final_index = constituents_step1
        else:
            constituents_step_1_2 = self.rebalance_process_2(index, 
                                                             df_month_year, 
                                                             constituents_step1)
            
            if len(constituents_step_1_2['Company Name']) == elements_index:
                final_index = constituents_step_1_2
                exit
            else:
                rest = elements_index - len(constituents_step_1_2['Company Name'])
                final_index  = self.rebalance_process_3(constituents_step_1_2, 
                                                        df_month_year, 
                                                        index, rest)
        
        final_index.sort_values(by=final_index.columns[-1], ascending=False, inplace = True)
        return final_index
    
    def portfolio_creation(self, index: pd.DataFrame, date: dt.date) -> pd.DataFrame:
        """
        Function which does all the process first rebalance process then optimize the weights of the stocks
        @params:
            index: Current index (DataFrame).
            date: Stocks from the starting universe for a particular (date)
    
        @outputs:
            final_index : The resulting index once the rebalance process and optimization
            of weights is done (DataFrame)
            minimization_total: The minimized sum of the objective function (int)
        """
        df_month_year = data[data['Ref Date']==date]
        index = self.main(index, df_month_year)    
        input_index = self.const_inputs(index)    
        self.constraints_creator(input_index)
        self.bounds_creator(input_index['Min Wt'].values, input_index['Max Wt'].values)
        final_weights = self.optimization_weigths(input_index['Min Wt'].values, 
                                                  input_index['Uncapped Wt'].values)
        index_final = self.index_opt_weigthed(index, final_weights.x)
        minimization_total = final_weights.fun
        return index_final, minimization_total
    
    
if __name__ == '__main__':
    
   inputs_path = r'C:\Users\Leonardo\Desktop\Personal\S&P500_test'
   inputs_file = r'Python assessment.xlsx'
   inputs_sheet_name = r'Start Universe'

       
   parameters = InputsDescription(inputs_path, inputs_file, inputs_sheet_name)  
    
   data = pd.read_excel(parameters.inputs_path + os.sep +
                         parameters.inputs_file,
                         sheet_name = parameters.inputs_sheet_name)
    
   data['Ref Date'] = data['Ref Date'].dt.date
    
   prt = Calculations(data)
    
   min_date = min(data['Ref Date'].unique())
   data_min = data[data['Ref Date']==min_date]
   index = prt.index_creation(data_min)
   portfolios_weighted = []
   minzation_value = []
   for date in data['Ref Date'].unique()[1:]:
       print(date)
       result = prt.portfolio_creation(index, date)
       print(type(result))
       portfolios_weighted.append(result[0])
       minzation_value.append(result[1])
       print(f'Minimization for {date} : {result[1]}')
       index = portfolios_weighted[-1]

with pd.ExcelWriter(parameters.inputs_path + os.sep +
                         parameters.inputs_file, engine='openpyxl') as writer:
    writer.book = load_workbook(r"Python assessment_test.xlsx")
    portfolios_weighted[0].to_excel(writer, sheet_name='Dic_2017_test_1_2', 
                               float_format="%.6f", index=False)
    portfolios_weighted[1].to_excel(writer, sheet_name='Jun_2018_test_1_2', 
                               float_format="%.6f", index=False)
    