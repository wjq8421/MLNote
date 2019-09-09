```shell
$ g++ -o first first.cpp
```

```#include <iostream>```：告诉编译器应该到当前编程环境的默认位置去检索相应的函数库。
```#include "somefile.h"```：编译器将到一个相对于当前文件的子目录里去寻找相应的函数库。

```std::cin.get()```：放在return前，使函数在输出信息后等待用户按下回车键。

```c++
char myChar;
std::cin.get(myChar); 	// 读入任意一个字符，包括空白字符
```

​	Q：`cin`对象从控制台读取输入，当用户按下回车键时，已经敲入的东西将一次性地被发送到C++程序。这样一来，当执行到`cin.get()`行时，将立刻读取那个回车而不是等你再次按下回撤键。

​	A：及时清除多余的（保存在缓冲区里的）输入数据。<font color=red> `std::cin.ignore(10, '\n')`</font>，其表示丢弃10个字符，或直到它遇到一个换行符。

​	`std::cin.gcount()`：返回缓冲区里的字符个数。



```c++
std::string input;
std::getline(std::cin, input);	//读取输入直到它遇到一个换行符
```

​	对输入数据进行合法性检查：

```c++
int age;
std::cout << "Enter your age: ";
while (!(std::cin >> age)) {
  // cin为假，或cin.fail()或cin.bad()调用的返回值是true：
  // 先调用clear函数清除cin的出错状态，再调用ignore()函数把缓冲区里的现有输入全部丢弃
	std::cin.clear();
  std::cin.ignore(100, '\n');
  std::cout << "Enter your age: ";
} // End of while

std::cin.ignore(100, '\n');
```



应该尽可能地使用4个空格而不是一个制表符。

______

```std::cout.precision(4);```：限定被输出的数值保留多少有效位数。

```c++
// 精确度设为小数点后面固定有2位
std::cout.setf(std::ios::fixed);
std::cout.setf(std::ios::showpoint);
std::cout.precision(2);
```

​	从实数到整数的类型转换不进行任何舍入处理，若想获得四舍五入的效果，需在转换前给那个实数加上0.5。

#### string

```c++
#include <string>
std::string firstName = "Larry";
firstName.size();
firstName == firstName2;	// 比较字符串是否相等
```

______



​	必须在声明一个常量的同时对它进行初始化。```const double PI = 3.1415926;```

#### cmath



______

​	应该坚持使用同一种代码排版格式。

​	三元操作符：```(condition) ? returnThisIfTrue : returnThisIfFalse;```

​	比较操作符：`&&`与，`||`或，`!`非

​	嵌套使用控制结构时需注意的事项：使用注释来说明某个控制结构的用途，表明它们的开始和结束位置。



```c++
sizeof(int);	//以字节为单位返回某给定变量或变量类型在当前操作系统上的内存占用量
sizeof myVar;
```

_____

#### cctype

```c++
toupper('a');
tolower('A');
isalnum('a');  // 检查字符是不是一个字母或数字
isalpha('a');
isdigit(10);
isspace(' ');		// 是不是一个空白字符
```

______

#### cstdlib

```c++
strtol();		// 把一个字符串转换为long，可安全地获得输入数据
strtod();		// 字符串 -> double
strtold();	// 字符串 ->long double
```

____

#### fstream

```c++
// 把数据输出到文件
std::ofstream fout("filename", std::ios::app);

if (fout.is_open()) {
  fout << "Hello World\n";
	//...
  fout.close(); 
}
///////////////////////

std::ifstream fin("filename");  // filename文件必须已经存在

if (fin.is_open()) {
  std::string line;
  while (std::getline(fin, line)) {
    //...
  } // End of while
  fin.close();
}
```

_____

​	编程习惯：在每个函数的结尾加上一条注释。坚持使用同一种风格的命名方式。

​	函数的原型通常集中安排在`main()`函数的定义之前，而定义函数的完整内容依次列在`main()`函数的后面。



​	给函数的输入参数设置一个默认值：**在函数原型里用赋值操作符把一个值赋值给那个输入参数**。且必须把所有的必选参数放在可选参数之前。

```c++
void fN(int n1, int n2=6);

// 使用注释将默认值标识出来
void fN(int n1, int n2 /* =6 */) {
  // ...
}
```

​	内联函数（inline function）：在`main()`函数的签名定义它，不需要先为它定义一个原型。



​	**函数重载**（Overloading）：用同样的名字再定义一个有着不同参数但有着同样用途的函数。重载的函数最适用于需要对不同的数据类型进行同样处理的场合。



​	强制转换操作符：`static_cast、reinterpret_cast、const_cast、dynamic_cast`，其语法是`operator<type>(data)`。

- `static_cast`：其和老式的C强制转换操作符的用法差不多，用来把一个简单的类型转换为另一个。
- `reinterpret_cast`：可以在不改变实际数据的情况下改变数据的类型，其易留下编程漏洞。但可用于把地址转换为一个整数。
- `const_cast`：把一个`const`类型的常量转换为一个非常量值。
- `dynamic_cast`：要与类搭配使用。



​	无类型指针：`void *pointerName`。



​	数组的名字同时也是一个指向其基地址（其第一个元素的地址）的指针。

```c++
int myArray[] = {25, 209, -12};
int *ptr1 = &myArray[0];
int *ptr2 = myArray;
ptr1++;	// 指向下一个元素的地址
```



​	联合（union）：可容纳多种不同类型的值，但它每次只能存储这些值中的某一个。

```c++
union id {
	std::string maidenName;
	unsigned short ssn;
	std::string pet;
};
```



​	以引用传递方式传递输入参数：

```c++
void changeVar(int &myVar, int newValue);
// 调用方式
int myNum = 20;
changeVar(myNum, 90);
// changeVar(20, 90); wrong

int &myFunction();	// 以引用传递方式返回

// 获得引用传递方式带来的性能改善，但不想改变其值
void myFunc(const int &myNum);
myFunc(7);	
```



​	类型别名：

```c++
typedef int* intPointer;
```

​	`enum`类型：1）对变量的可取值加以限制；2）可用作`switch`条件语句的`case`标号。

```c++
enum weekdays {Monday, Tuesday, Wednesday, Thursday, Friday};
weekdays today;
today = Tuesday;
```



#### 基本概念

##### 类

​	C++允许在类里声明常量，但不允许对它进行赋值。若想绕开这一限制，可创建一个静态常量。

```c++
class Car {
	public:
		static const float TANKSIZE = 12.5;
};
```



​	如果一个类的构造器从计算机申请了一块内存，就必须在析构器里释放那块内存。

