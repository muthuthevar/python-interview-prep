# Object-Oriented Programming & Design Patterns

## OOP Principles

### 1. The Four Pillars of OOP

#### Encapsulation
- Bundling data and methods together
- Access modifiers (public, protected, private)
- Python's convention: `_protected`, `__private`
- Property decorators for controlled access

#### Inheritance
- Single and multiple inheritance
- Method Resolution Order (MRO)
- `super()` function
- Method overriding vs overloading

#### Polymorphism
- Duck typing
- Operator overloading
- Method overriding
- Abstract base classes

#### Abstraction
- Abstract classes (`abc` module)
- Abstract methods
- Interface-like behavior in Python

### 2. Class Design

#### Class vs Instance Variables
```python
class MyClass:
    class_var = "shared"  # Class variable
    
    def __init__(self):
        self.instance_var = "unique"  # Instance variable
```

#### Class Methods vs Static Methods
```python
class MyClass:
    @classmethod
    def class_method(cls):
        # Receives class as first argument
        # Can access class variables
        pass
    
    @staticmethod
    def static_method():
        # No implicit first argument
        # Cannot access class or instance
        pass
```

#### Property Decorators
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2
```

### 3. Special Methods (Magic Methods)

#### Common Magic Methods
- `__init__`: Constructor
- `__str__`: String representation (user-friendly)
- `__repr__`: String representation (developer-friendly)
- `__eq__`, `__ne__`: Equality comparison
- `__lt__`, `__le__`, `__gt__`, `__ge__`: Comparison operators
- `__hash__`: Hash function
- `__len__`: Length
- `__getitem__`, `__setitem__`: Indexing
- `__iter__`, `__next__`: Iterator protocol
- `__call__`: Make instance callable
- `__enter__`, `__exit__`: Context manager

#### Example: Custom Class with Magic Methods
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)
```

## Design Patterns

### 1. Creational Patterns

#### Singleton Pattern
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Using decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance
```

#### Factory Pattern
```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")
```

#### Builder Pattern
```python
class Pizza:
    def __init__(self):
        self.size = None
        self.cheese = False
        self.pepperoni = False
        self.bacon = False
    
    def __str__(self):
        return f"Pizza(size={self.size}, cheese={self.cheese}, pepperoni={self.pepperoni}, bacon={self.bacon})"

class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()
    
    def set_size(self, size):
        self.pizza.size = size
        return self
    
    def add_cheese(self):
        self.pizza.cheese = True
        return self
    
    def add_pepperoni(self):
        self.pizza.pepperoni = True
        return self
    
    def add_bacon(self):
        self.pizza.bacon = True
        return self
    
    def build(self):
        return self.pizza

# Usage
pizza = PizzaBuilder().set_size("large").add_cheese().add_pepperoni().build()
```

### 2. Structural Patterns

#### Adapter Pattern
```python
class OldSystem:
    def old_method(self):
        return "Old interface"

class Adapter:
    def __init__(self, old_system):
        self.old_system = old_system
    
    def new_method(self):
        return self.old_system.old_method()
```

#### Decorator Pattern
```python
class Component:
    def operation(self):
        return "Component"

class Decorator(Component):
    def __init__(self, component):
        self.component = component
    
    def operation(self):
        return f"Decorator({self.component.operation()})"

class ConcreteDecorator(Decorator):
    def operation(self):
        return f"ConcreteDecorator({self.component.operation()})"
```

#### Facade Pattern
```python
class Subsystem1:
    def operation1(self):
        return "Subsystem1 operation"

class Subsystem2:
    def operation2(self):
        return "Subsystem2 operation"

class Facade:
    def __init__(self):
        self.subsystem1 = Subsystem1()
        self.subsystem2 = Subsystem2()
    
    def simplified_operation(self):
        result1 = self.subsystem1.operation1()
        result2 = self.subsystem2.operation2()
        return f"{result1} + {result2}"
```

### 3. Behavioral Patterns

#### Observer Pattern
```python
class Observer:
    def update(self, message):
        pass

class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name
    
    def update(self, message):
        print(f"{self.name} received: {message}")
```

#### Strategy Pattern
```python
class PaymentStrategy:
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} using Credit Card"

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} using PayPal"

class PaymentContext:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def execute_payment(self, amount):
        return self.strategy.pay(amount)
```

#### Command Pattern
```python
class Command:
    def execute(self):
        pass

class Light:
    def turn_on(self):
        return "Light is ON"
    
    def turn_off(self):
        return "Light is OFF"

class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_on()

class LightOffCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_off()

class RemoteControl:
    def __init__(self):
        self.command = None
    
    def set_command(self, command):
        self.command = command
    
    def press_button(self):
        return self.command.execute()
```

## Common Interview Questions

### Q1: Explain the difference between `__new__` and `__init__`
- `__new__`: Creates the instance (class method, returns instance)
- `__init__`: Initializes the instance (instance method, returns None)
- `__new__` is called before `__init__`

### Q2: What is method resolution order (MRO)?
- Order in which base classes are searched for a method
- Uses C3 linearization algorithm
- Can be viewed with `ClassName.__mro__`
- Important for multiple inheritance

### Q3: Explain `super()` function
- Returns a proxy object that delegates method calls to parent class
- Handles MRO automatically
- Useful in multiple inheritance scenarios

### Q4: What are abstract base classes?
- Classes that cannot be instantiated
- Define interface that subclasses must implement
- Use `abc` module: `ABC`, `abstractmethod`

### Q5: Explain composition vs inheritance
- Inheritance: "is-a" relationship
- Composition: "has-a" relationship
- Favor composition over inheritance (more flexible)

### Q6: What is duck typing?
- "If it walks like a duck and quacks like a duck, it's a duck"
- Focus on behavior, not type
- Python's dynamic typing philosophy

### Q7: Explain the difference between `@staticmethod` and `@classmethod`
- `@staticmethod`: No implicit first argument, cannot access class/instance
- `@classmethod`: Receives class as first argument, can access class variables

### Q8: What is the difference between `__str__` and `__repr__`?
- `__str__`: User-friendly representation
- `__repr__`: Unambiguous representation (ideally valid Python code)
- `print()` uses `__str__`, REPL uses `__repr__`

## Design Principles

### SOLID Principles

#### Single Responsibility Principle
- A class should have only one reason to change
- Each class should have one job

#### Open/Closed Principle
- Open for extension, closed for modification
- Use inheritance and polymorphism

#### Liskov Substitution Principle
- Subtypes must be substitutable for their base types
- Derived classes should not break base class contracts

#### Interface Segregation Principle
- Clients should not depend on interfaces they don't use
- Prefer many specific interfaces over one general interface

#### Dependency Inversion Principle
- Depend on abstractions, not concretions
- High-level modules should not depend on low-level modules

### DRY (Don't Repeat Yourself)
- Avoid code duplication
- Extract common functionality

### KISS (Keep It Simple, Stupid)
- Simplicity should be a key goal
- Avoid unnecessary complexity

### YAGNI (You Aren't Gonna Need It)
- Don't add functionality until it's necessary
- Avoid over-engineering

## Practice Problems

### Problem 1: Design a Parking Lot System
- Multiple parking spots
- Different vehicle types (car, motorcycle, truck)
- Assign spots based on vehicle type
- Track available spots

### Problem 2: Design a Library Management System
- Books, Members, Transactions
- Checkout, return, reserve functionality
- Fine calculation
- Search functionality

### Problem 3: Design a Restaurant Ordering System
- Menu items
- Orders
- Payment processing
- Kitchen queue management

### Problem 4: Implement a Cache with LRU Eviction
- Fixed size cache
- Get and put operations
- LRU (Least Recently Used) eviction policy

### Problem 5: Design a File System
- Files and directories
- Path navigation
- Create, delete, move operations
- Size calculation
