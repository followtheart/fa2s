# FA2S: Finite Automaton with 2 Stacks

FA2S 是一门以“有穷自动机 + 两个无界栈（A、B）”为计算模型的极简语言。它以“状态-转移-动作”的自动机风格编程，提供：
- 输入流读取（read）、输出（write）
- 两个栈的基本操作（push/pop），以及把栈顶转存到寄存器（last）
- 基于输入/栈顶的守卫选择不同转移（确定式，按顺序匹配第一条）

由于两个栈等价图灵机，FA2S 具备图灵完备的计算能力。

## 程序模型

- 状态：有限集合的命名状态。程序从 `.start <state>` 指定的初始状态开始（默认 `start`）。
- 输入：只读字符流，指针从 0 前进到 EOF。`read` 将当前位置字符读入寄存器 `last` 并前移指针。
- 输出：`write:"..."` 写字符串；`write:last` 写寄存器 `last` 的单字符。
- 栈：两个无界栈 A、B，初始为空。
- 转移选择：在当前状态，按源代码书写顺序匹配“守卫”（输入模式 + 栈顶模式）。命中第一条后，执行其动作列表，再跳转到下一状态或终止。
- 终止：`halt:accept` 或 `halt:reject` 立即停止。

## 语法

一行一条转移或指令，`#` 为注释，空行忽略。

指令
- `.start <stateName>` 指定初始状态（可省略，默认 `start`）。

转移（单行）
```
<state> : <IN>, A=<ATOP>, B=<BTOP> -> <action1>[, <action2> ...] ; <nextState>
```
- 若使用 `halt:accept` / `halt:reject`，则不需要 `; <nextState>`。

守卫（Guard）
- `IN` ∈ { `eps`, `eof`, `any`, `lit:'x'` }
  - `eps`：不读输入即可触发
  - `eof`：仅在已到达输入末尾
  - `any`：当前位置有任意字符（尚未 EOF）
  - `lit:'x'`：当前位置字符为 x（单字符，支持转义如 `'\n'`）
- `ATOP` / `BTOP` ∈ { `*`, `empty`, `lit:'x'` }
  - `*`：任意栈顶（包括空）
  - `empty`：栈为空
  - `lit:'x'`：栈顶是字符 x

动作（按顺序执行）
- 输入/输出
  - `read`：读当前位置字符到寄存器 `last` 并前移指针（若已 EOF，运行时错误）
  - `write:"string"`：输出字符串（双引号内可用 `\n \t \\ \"` 转义）
  - `write:last`：输出寄存器 `last` 单字符（需此前有 `read` 或 `popX:last`）
- 栈操作
  - `pushA:'x'` / `pushB:'x'`：压入字符 x
  - `pushA:last` / `pushB:last`：压入寄存器 `last` 中的字符
  - `popA` / `popB`：弹栈（丢弃，空栈报错）
  - `popA:last` / `popB:last`：弹栈并存入寄存器 `last`
- 终止
  - `halt:accept` | `halt:reject`

语义要点
- 转移按书写顺序匹配“第一条命中”的规则。
- `read` 与输入守卫无强制耦合；你可以在 `eps` 守卫里 `read`，也可以在 `lit:'x'` 守卫里不 `read`。推荐写法是：若守卫关心当前字符，就紧随其后 `read` 以消耗该字符。
- 为避免死循环，解释器默认有步数上限（可传参调整）。

## 示例

1) 回显（echo）
逐字符读取并立即输出：
```
.start start

start : eps, A=empty, B=empty -> ; loop

loop  : any, A=*, B=* -> read, write:last ; loop
loop  : eof, A=*, B=* -> halt:accept
```

2) 反向输出（reverse）
先把输入全部压入 A，再逐个弹出并输出：
```
.start start

start     : eps, A=empty, B=empty -> ; read_all
read_all  : any, A=*, B=* -> read, pushA:last ; read_all
read_all  : eof, A=*, B=* -> ; write_back

write_back: eps, A=empty, B=* -> halt:accept
write_back: eps, A=*,     B=* -> popA:last, write:last ; write_back
```

3) 括号匹配（仅 '(' 和 ')'）
遇到 '(' 入栈，遇到 ')' 弹栈；其他字符忽略。EOF 时空栈即接受。
```
.start start

start : eps, A=empty, B=empty -> ; loop

# 读 '(' 入栈
loop  : lit:'(', A=*,        B=* -> read, pushA:last ; loop
# 读 ')' 且匹配栈顶 '(' 则弹栈
loop  : lit:')', A=lit:'(',  B=* -> read, popA ; loop
# 读 ')' 但栈空 -> 拒绝
loop  : lit:')', A=empty,    B=* -> read, halt:reject
# 其他字符：忽略
loop  : any,     A=*,        B=* -> read ; loop

# EOF 时空栈接受，否则拒绝
loop  : eof,     A=empty,    B=* -> write:"ok\n",  halt:accept
loop  : eof,     A=*,        B=* -> write:"err\n", halt:reject
```

## 运行

```
python3 src/fa2s.py examples/echo.fa2s --input "hello"
python3 src/fa2s.py examples/reverse.fa2s --input "abcd"
python3 src/fa2s.py examples/balparen.fa2s --input "(())()"
```

选项
- `--input "<str>"`：直接提供输入字符串（否则读 stdin）
- `--infile <path>`：从文件读取输入
- `--max-steps N`：步数上限（默认 1_000_000）
- `--trace`：打印执行轨迹（调试用）

## 设计与可扩展点

- 已图灵完备：两个无界栈 + 有限控制即等价图灵机。你可以在两个栈间编码队列、带状内存或寄存器机，从而实现一般计算。
- 可选扩展（向后兼容）：
  - 增加字符类匹配（如 `class:"abc"`、范围 `range:"a-z0-9"`）
  - 增加宏/子程序（状态前缀 + 返回约定）
  - 增加“非确定分支”（目前通过多条可命中规则 + 顺序优先模拟；也可加入 `choose` 指令）

## 错误与诊断

- 没有命中任何转移：运行时错误（无定义行为）。
- 空栈弹出、EOF 时 `read`：运行时错误。
- 可用 `--trace` 查看每步状态、指针、栈顶与动作。