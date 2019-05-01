    
class duck():
    count = 0

    def __init__(self, input_name):
        duck.count += 1
        self.hidden_name = input_name
    
    def k(self,a,b):
        return a+b

    def g(self,c,a,b):
        return c+self.k(a,b)


    @property
    def name(self):
        print('inside the getter')

    @name.setter
    def name(self, input_name):
        print('inside the setter')
        self.hidden_name = input_name

    @classmethod
    def kids(clsss):
        print('duck has', clsss.count, 'little objects')


if __name__ == '__main__':
    a = duck('a')
    b = duck('b')
    b = duck('b')
    #duck.kids()
    #print(duck.count)
    print(b.g(1,2,3))