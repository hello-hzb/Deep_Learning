# -*- coding: utf-8 -*-
import math
import time


class Circle:
    def __init__(self, radius):           # 圆的半径radius
        self.radius = radius

    # @property
    def area(self):
        return math.pi * self.radius**2   # 计算面积

    # @property
    def perimeter(self):
        return 2*math.pi*self.radius      # 计算周长

c = Circle(10)
print(c.radius)
print(c.area())                    # 可以向访问数据属性一样去访问area,会触发一个函数的执行,动态计算出一个值
print(c.perimeter)                  # 同上
'''
输出结果:
314.1592653589793
62.83185307179586
'''


class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    @staticmethod
    def now():                 # 用Date.now()的形式去产生实例,该实例用的是当前时间
        t = time.localtime()   # 获取结构化的时间格式
        return Date(t.tm_year, t.tm_mon, t.tm_mday)  # 新建实例并且返回

    @staticmethod
    def tomorrow():  # 用Date.tomorrow()的形式去产生实例,该实例用的是明天的时间
        t = time.localtime(time.time()+86400)
        return Date(t.tm_year, t.tm_mon, t.tm_mday)


a = Date('1987', 11, 27)  # 自己定义时间
b = Date.now()            # 采用当前时间
c = Date.tomorrow()       # 采用明天的时间

print(a.year, a.month, a.day)
print(b.year, b.month, b.day)
print(c.year, c.month, c.day)



class A(object):
    bar = 1
    def foo(self):
        print ('foo')

    @staticmethod
    def static_foo():
        print ('static_foo')
        print (A.bar)

    @classmethod
    def class_foo(cls):
        print ('class_foo')
        print (cls.bar)
        cls().foo()
###执行
A.static_foo()
A.class_foo()


