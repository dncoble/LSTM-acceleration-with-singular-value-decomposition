class One():
    
    def __init__(self):
        print("one")
    
    def prt(self):
        print("one")

class Two(One):
    
    def __init__(self):
        print("two")
        
    
    def prt(self):
        print("two")

class Has():
    
    def __init__(self):
        self.number = One()
    
    
has = Has()

has.number = Two()

has.number.prt()

"""
one
two
one
"""
