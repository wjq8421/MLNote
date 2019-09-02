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

