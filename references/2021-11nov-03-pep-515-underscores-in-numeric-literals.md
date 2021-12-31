# PEP 515 -- Underscores in Numeric Literals
Georg Brandl - 2016 (3 minute read) [View Original](https://www.python.org/dev/peps/pep-0515/)

> The official home of the Python Programming Language

**Notice:** While JavaScript is not essential for this website, your interaction with the content will be limited. Please turn JavaScript on for the full experience.

<table><colgroup><col> <col></colgroup><tbody><tr><th>PEP:</th><td>515</td></tr><tr><th>Title:</th><td>Underscores in Numeric Literals</td></tr><tr><th>Author:</th><td>Georg Brandl, Serhiy Storchaka</td></tr><tr><th>Status:</th><td>Final</td></tr><tr><th>Type:</th><td>Standards Track</td></tr><tr><th>Created:</th><td>10-Feb-2016</td></tr><tr><th>Python-Version:</th><td>3.6</td></tr><tr><th>Post-History:</th><td>10-Feb-2016, 11-Feb-2016</td></tr></tbody></table>

* * *

This PEP proposes to extend Python's syntax and number-from-string constructors so that underscores can be used as visual separators for digit grouping purposes in integral, floating-point and complex number literals.

This is a common feature of other modern languages, and can aid readability of long literals, or literals whose value should clearly separate into parts, such as bytes or words in hexadecimal notation.

Examples:

\# grouping decimal numbers by thousands
amount = 10\_000\_000.0

# grouping hexadecimal addresses by words
addr = 0xCAFE\_F00D

# grouping bits into nibbles in a binary literal
flags = 0b\_0011\_1111\_0100\_1110

# same, for string conversions
flags = int('0b\_1111\_0000', 2)

The current proposal is to allow one underscore between digits, and after base specifiers in numeric literals. The underscores have no semantic meaning, and literals are parsed as if the underscores were absent.

[Literal Grammar](#id28)
------------------------

The production list for integer literals would therefore look like this:

integer: decinteger | bininteger | octinteger | hexinteger
decinteger: nonzerodigit (\["\_"\] digit)\* | "0" (\["\_"\] "0")\*
bininteger: "0" ("b" | "B") (\["\_"\] bindigit)+
octinteger: "0" ("o" | "O") (\["\_"\] octdigit)+
hexinteger: "0" ("x" | "X") (\["\_"\] hexdigit)+
nonzerodigit: "1"..."9"
digit: "0"..."9"
bindigit: "0" | "1"
octdigit: "0"..."7"
hexdigit: digit | "a"..."f" | "A"..."F"

For floating-point and complex literals:

floatnumber: pointfloat | exponentfloat
pointfloat: \[digitpart\] fraction | digitpart "."
exponentfloat: (digitpart | pointfloat) exponent
digitpart: digit (\["\_"\] digit)\*
fraction: "." digitpart
exponent: ("e" | "E") \["+" | "-"\] digitpart
imagnumber: (floatnumber | digitpart) ("j" | "J")

[Constructors](#id29)
---------------------

Following the same rules for placement, underscores will be allowed in the following constructors:

*   int() (with any base)
*   float()
*   complex()
*   Decimal()

[Further changes](#id30)
------------------------

The new-style number-to-string formatting language will be extended to allow \_ as a thousands separator, where currently only , is supported. This can be used to easily generate code with more readable literals. [\[11\]](#id24)

The syntax would be the same as for the comma, e.g. {:10\_} for a width of 10 with \_ separator.

For the b, x and o format specifiers, \_ will be allowed and group by 4 digits.

Those languages that do allow underscore grouping implement a large variety of rules for allowed placement of underscores. In cases where the language spec contradicts the actual behavior, the actual behavior is listed. ("single" or "multiple" refer to allowing runs of consecutive underscores.)

*   Ada: single, only between digits [\[8\]](#id21)
*   C# (open proposal for 7.0): multiple, only between digits [\[6\]](#id19)
*   C++14: single, between digits (different separator chosen) [\[1\]](#id14)
*   D: multiple, anywhere, including trailing [\[2\]](#id15)
*   Java: multiple, only between digits [\[7\]](#id20)
*   Julia: single, only between digits (but not in float exponent parts) [\[9\]](#id22)
*   Perl 5: multiple, basically anywhere, although docs say it's restricted to one underscore between digits [\[3\]](#id16)
*   Ruby: single, only between digits (although docs say "anywhere") [\[10\]](#id23)
*   Rust: multiple, anywhere, except for between exponent "e" and digits [\[4\]](#id17)
*   Swift: multiple, between digits and trailing (although textual description says only "between digits") [\[5\]](#id18)

[Underscore Placement Rules](#id33)
-----------------------------------

Instead of the relatively strict rule specified above, the use of underscores could be less limited. As seen in other languages, common rules include:

*   Only one consecutive underscore allowed, and only between digits.
*   Multiple consecutive underscores allowed, but only between digits.
*   Multiple consecutive underscores allowed, in most positions except for the start of the literal, or special positions like after a decimal point.

The syntax in this PEP has ultimately been selected because it covers the common use cases, and does not allow for syntax that would have to be discouraged in style guides anyway.

A less common rule would be to allow underscores only every N digits (where N could be 3 for decimal literals, or 4 for hexadecimal ones). This is unnecessarily restrictive, especially considering the separator placement is different in different cultures.

[Different Separators](#id34)
-----------------------------

A proposed alternate syntax was to use whitespace for grouping. Although strings are a precedent for combining adjoining literals, the behavior can lead to unexpected effects which are not possible with underscores. Also, no other language is known to use this rule, except for languages that generally disregard any whitespace.

C++14 introduces apostrophes for grouping (because underscores introduce ambiguity with user-defined literals), which is not considered because of the use in Python's string literals. [\[1\]](#id14)

A preliminary patch that implements the specification given above has been posted to the issue tracker. [\[12\]](#id25)

This document has been placed in the public domain.

Source: [https://github.com/python/peps/blob/master/pep-0515.txt](https://github.com/python/peps/blob/master/pep-0515.txt)


[Source](https://www.python.org/dev/peps/pep-0515/)