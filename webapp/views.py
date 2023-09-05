from django.shortcuts import render,redirect
import math
from django.http import JsonResponse 
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from scipy.stats import norm
import numpy as np    

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def fetchData(ticker = "INFY.NS"):
    
    data = pdr.get_data_yahoo(ticker)
    data_AdjClose = data["Adj Close"]
    data_diff = data_AdjClose.diff()
    data_ret = data_diff/data_AdjClose
    ret_pos =[]
    ret_neg =[]
    for dt in data_ret:
        if dt >0:
            ret_pos.append(dt)
        elif dt < 0:
            ret_neg.append(dt)
        else:
            pass
    u = np.average(ret_pos)
    d = np.average(ret_neg)						
    data_close = data["Close"]
    s0 = data_close[-1]
    sigma = np.std(data_ret)
    return u, d, s0, sigma

class BinomialModel(object): 
    def __init__(self, s0, u, d, strike, maturity, rfr,  n, compd = "s", dyield = None):
        self.s0 = s0
        self.u = u
        self.d = d
        self.rfr = rfr
        self.maturity = maturity 
        self.strike = strike
        self.n = n
        self.compd = compd
        self.dyield = dyield
    
    def call_price(self):
        delta = float(self.maturity)/float(self.n)
        
        if self.compd == "c":
            if self.dyield ==None: 
                q = (math.exp(self.rfr*delta) - self.d) / (self.u - self.d)
            else:
                q = (math.exp((self.rfr-self.dyield)*delta) - self.d) / (self.u - self.d)
        if self.compd == "s":
            if self.dyield == None: 
                q = (1 + self.rfr*delta - self.d) / (self.u - self.d)
            else:
                q = (1+ (self.rfr - self.dyield)*delta - self.d) / (self.u - self.d)
        
        prc = 0
        temp_stock = 0
        temp_payout = 0
        for x in range(0, self.n + 1):
            temp_stock = self.s0*((self.u)**(x))*((self.d)**(self.n - x))
            temp_payout = max(temp_stock - self.strike, 0)
            prc += nCr(self.n, x)*(q**(x))*((1-q)**(self.n - x))*temp_payout
        
        if self.compd == "s":
            prc = prc / ((1+ self.rfr*delta )**self.n)
        if self.compd == "c":
            prc = prc / math.exp(self.rfr*delta)
        
        
        return prc
    
    
    def put_price(self):
        delta = float(self.maturity)/float(self.n)
        
        if self.compd == "c":
            if self.dyield ==None: 
                q = (math.exp(self.rfr*delta) - self.d) / (self.u - self.d)
            else:
                q = (math.exp((self.rfr-self.dyield)*delta) - self.d) / (self.u - self.d)
        if self.compd == "s":
            if self.dyield == None: 
                q = (1 + self.rfr*delta - self.d) / (self.u - self.d)
            else:
                q = (1+ (self.rfr - self.dyield)*delta - self.d) / (self.u - self.d)
        
        prc = 0
        temp_stock = 0
        temp_payout = 0
        for x in range(0, self.n + 1):
            temp_stock = self.s0*((self.u)**(x))*((self.d)**(self.n - x))
            temp_payout = max(self.strike - temp_stock, 0)
            prc += nCr(self.n, x)*(q**(x))*((1-q)**(self.n - x))*temp_payout
        
        if self.compd == "s":
            prc = prc / ((1+ self.rfr*delta )**self.n)
        if self.compd == "c":
            prc = prc / math.exp(self.rfr*delta)
        
        
        return prc
    
class CRRModel(object):
    def __init__(self, s0, sigma, strike, maturity, rfr,  n, compd = "s", dyield = None):      
        self.s0 = s0 
        self.sigma = sigma
        self.rfr = rfr
        self.maturity = maturity 
        self.strike = strike
        self.n = n
        self.compd = compd
        self.dyield = dyield
    
    def call_price(self):
        delta = float(self.maturity)/float(self.n)
        u = math.exp(self.sigma*math.sqrt(delta))
        d = 1/math.exp(self.sigma*math.sqrt(delta))


        if self.compd == "c":
            if self.dyield ==None: 
                q = (math.exp(self.rfr*delta) - d) / (u - d)
            else:
                q = (math.exp((self.rfr-self.dyield)*delta) - d) / (u - d)
        if self.compd == "s":
            if self.dyield == None: 
                q = (1 + self.rfr*delta - d) / (u - d)
            else:
                q = (1+ (self.rfr - self.dyield)*delta - d) / (u - d)
        
        prc = 0
        temp_stock = 0
        temp_payout = 0
        for x in range(0, self.n + 1):
            temp_stock = self.s0*((u)**(x))*((d)**(self.n - x))
            temp_payout = max(temp_stock - self.strike, 0)
            prc += nCr(self.n, x)*(q**(x))*((1-q)**(self.n - x))*temp_payout
        
        if self.compd == "s":
            prc = prc / ((1+ self.rfr*delta )**self.n)
        if self.compd == "c":
            prc = prc / math.exp(self.rfr*delta)
        
        
        return prc
    
    
    def put_price(self):
        delta = float(self.maturity)/float(self.n)
        u = math.exp(self.sigma*math.sqrt(delta))
        d= 1/math.exp(self.sigma*math.sqrt(delta))


        if self.compd == "c":
            if self.dyield ==None: 
                q = (math.exp(self.rfr*delta) - d) / (u - d)
            else:
                q = (math.exp((self.rfr-self.dyield)*delta) - d) / (u - d)
        if self.compd == "s":
            if self.dyield == None: 
                q = (1 + self.rfr*delta - d) / (u - d)
            else:
                q = (1+ (self.rfr - self.dyield)*delta - d) / (u - d)
        
        prc = 0
        temp_stock = 0
        temp_payout = 0
        for x in range(0, self.n + 1):
            temp_stock = self.s0*((u)**(x))*((d)**(self.n - x))
            temp_payout = max(self.strike - temp_stock, 0)
            prc += nCr(self.n, x)*(q**(x))*((1-q)**(self.n - x))*temp_payout
        
        if self.compd == "s":
            prc = prc / ((1+ self.rfr*delta )**self.n)
        if self.compd == "c":
            prc = prc / math.exp(self.rfr*delta)
        
        
        return prc  

class BlackScholeModel(object):
    def __init__(self, S, K, T, r, sigma, q=0,):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    @staticmethod
    def N(x):
        return norm.cdf(x)
    
    @property
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r': self.r,
                'q': self.q,
                'sigma': self.sigma}
    
    def d1(self):
        return (np.log(self.S / self.K) + (self.r - self.q + self.sigma**2 / 2) * self.T) \
                                / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def _call_value(self):
        return self.S * np.exp(-self.q * self.T) * self.N(self.d1()) - \
                    self.K * np.exp(-self.r * self.T) * self.N(self.d2())
                    
    def _put_value(self):
        return self.K * np.exp(-self.r * self.T) * self.N(-self.d2()) - \
                self.S * np.exp(-self.q * self.T) * self.N(-self.d1())
    
    def price(self, type_='C'):
        if type_ == 'C':
            return self._call_value()
        if type_ == 'P':
            return self._put_value() 
        if type_ == 'B':
            return {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Unrecognized type')

def getPriceAndProb(request):
    if request.method=='POST':
        initialEP=float(request.POST.get('val'))
        upFactor=float(request.POST.get('upFactor'))
        downFactor=float(request.POST.get('downFactor'))
        strikePrice=float(request.POST.get('strikePrice'))
        riskFreeRate=float(request.POST.get('riskFreeRate'))
        maturity=float(request.POST.get('maturity'))
        noOfPeriods=int(request.POST.get('noOfPeriods'))
        dYield=float(request.POST.get('dYield'))
        com=(request.POST.get('interest'))
        isPut=request.POST.get('isPut')
        model=BinomialModel(initialEP,upFactor,downFactor,strikePrice,maturity,riskFreeRate,noOfPeriods,compd=com,dyield=dYield)
        # fairPrice=model.call_price()
        # print(fairPrice)
        # fairPrice=model.put_price()
        # return JsonResponse({'fairPrice':fairPrice})
        # print(isPut==True)
        if isPut=='true':
            fairPrice=model.put_price()
            return JsonResponse({'fairPrice':round(fairPrice,2)})
        else:
            callFairPrice=model.call_price()
            return JsonResponse({'fairPrice':round(callFairPrice,2)})


def home(req):
    return render(req,'home.html',{})