class Value:
  #initialisation
  def __init__(self, data:float, _prev=(),_op=""):
    self.data=data
    self.grad=0
    self._prev=set(_prev)
    self._op=_op
    self._backward=lambda:None

  #representation of class
  def __repr__(self):
    return f"Data={self.data}, Grad={self.grad}"

  #core funcs
  def __add__(self, other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data+other.data,(self,other),"+")

    def _backward():
      self.grad+=(out.grad*1.0)
      other.grad+=(out.grad*1.0)
    out._backward=_backward # Assign the backward function to out
    return out

  def __mul__(self,other):
    other= other if isinstance(other,Value) else Value(other)
    out = Value(self.data*other.data,(self,other),"*")

    def _backward():
      self.grad+= (other.data*out.grad)
      other.grad+= (self.data*out.grad)
    out._backward=_backward # Assign the backward function to out
    return out

  def __pow__(self,other):
    assert isinstance(other, (float,int )), "required float/int power"
    out= Value((self.data)**other,(self,),"^")

    def _backward():
      self.grad+= out.grad*other*(self.data**(other-1))
    out._backward=_backward # Assign the backward function to out
    return out

  #derived
  def __neg__(self):
    return self*-1

  def __sub__(self,other):
    return self+ (-other)

  def __truediv__(self,other):
    return self*(other**-1)

  #reverse func
  def __radd__(self,other):
    return self+other

  def __rsub__(self,other):
    return other+(-self)

  def __rmul__(self,other):
    return self*other

  def __rtruediv__(self,other):
    return other*(self**-1)

  #activation function
  def relu(self):
    out= Value(max(0,self.data),(self,),"relu")

    def _backward():
      self.grad+= (out.grad if self.data>0 else 0.0)
    out._backward=_backward # Assign the backward function to out
    return out

  #back propogation: perform topological sort, then traverse in reverse topological order
  def backward(self):
    #topo-sort
    topo=[]
    vis=set()
    def dfs(node):
      if node not in vis:
        vis.add(node)
        for neighbour in node._prev:
          dfs(neighbour)
        topo.append(node)
    dfs(self)
    #grad calc
    self.grad=1
    for node in reversed(topo):
      node._backward()