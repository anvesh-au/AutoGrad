import random
class Module:
  def parameters(self):
    return []

  def zero_grad(self):
    for p in self.parameters():
      p.grad=0

class Neuron(Module):
  def __init__(self,nin:int,nonlin:bool):
    self.w=[Value(random.uniform(-1,1) ) for _ in range(nin)]
    self.b=Value(0)
    self.nonlin=nonlin

  def parameters(self):
    return self.w+[self.b]

  def __call__(self,x):
    o=sum(w_i*x_i for (w_i,x_i) in zip(self.w,x))+ (self.b)
    return o.relu() if self.nonlin else o

  def __repr__(self):
    return f"Activation={"ReLu" if self.nonlin else "Linear"}, Weights={self.parameters()}"

class Layer(Module):
  def __init__(self, nin, nout, **kwargs):
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

  def __call__(self,x):
    o=[neuron(x) for neuron in self.neurons]
    return o[0] if len(o)==1 else o

  def __repr__(self):
    return f"Layer of [{"\n".join(str(neuron) for neuron in self.neurons)}]"

class MLP(Module):
  def __init__(self,nin:int,nouts:list):
    sz=[nin]+nouts
    self.layers=[Layer(sz[i],sz[i+1],nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]

  def __repr__(self):
    return f"Neural net of layers [{"\n".join(str(layer) for layer in self.layers)}]"

  def __call__(self,x):
    for layer in self.layers:
      x=layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]