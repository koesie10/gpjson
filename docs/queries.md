# Supported queries

Query format is JSONPath, with the following limitations:

- No deep scan operator (`..`)
- No expressions (`?(<expression>)`)
- No multiple array indexes (`[<number>, <number> (, <number>)]`)
- No functions (`min`, `max`, etc.)

So, from the examples given in [JsonPath](https://github.com/json-path/JsonPath#path-examples), the following are
supported:

| JsonPath | Result |
| :------- | :----- |
| $.store.book[*].author| The authors of all books     |
| ~~$..author~~                   | All authors                         |
| $.store.*                  | All things, both books and bicycles  |
| ~~$.store..price~~             | The price of everything         |
| $..book[2]                 | The third book                      |
| $..book[-2]                 | The second to last book            |
| $..book[0,1]               | The first two books               |
| $..book[:2]                | All books from index 0 (inclusive) until index 2 (exclusive) |
| $..book[1:2]                | All books from index 1 (inclusive) until index 2 (exclusive) |
| $..book[-2:]                | Last two books                   |
| $..book[2:]                | Book number two from tail          |
| ~~$..book\[?(@.isbn)]~~          | All books with an ISBN number         |
| ~~$.store.book\[?(@.price < 10)]~~ | All books in store cheaper than 10  |
| ~~$..book[?(@.price <= $\['expensive'])]~~ | All books in store that are not "expensive"  |
| ~~$..book\[?(@.author =~ /.*REES/i)]~~ | All books matching regex (ignore case)  |
| ~~$..*~~                        | Give me every thing
| ~~$..book.length()~~                 | The number of books                      |

Some notes regarding JSONPath:
- When using an index expression, this does not match numeric keys. So, if instead of the example on the page mentioned
before, the `book` array is an object of the form `{"0": {...}, "1": {...}, "2": {...}, "3": {...}}`, the expression
`$..book[2]` matches nothing. `$..book.2` and `$..book['2']` do match the correct key, but doesn't work on indices.
- When a dot `.` is used, it is ignored unless it's repeated `..`, so `$..book['2']` and `$..book.['2']` are identical.

# Intermediate representation

The root token `$` is irrelevant, so it is not present in the IR. The relevant types of subpaths are shown below. Paths
with an identical name are identical.

| JsonPath | Name |
| :------- | :--- |
| `store`  | Property |
| `['store']` | Property |
| `['store', 'expensive']` | Multiple property |
| `[1]` | Index array |
| `[1, 2]` | Multiple index array |
| `[1:2]` | Slice array|
| `[*]` | Wildcard |
| `*` | Wildcard |

The property and multiple property and index array and multiple index array can be combined, so that results in the
following token types:
- Property
- Index array
- Slice array
- Wildcard

We probably want to know the types more often than the actual contents, so it is better to group the types together. To
allow easy access, each type will be represented as 1 byte. We will reserve 8 bytes for any additional metadata (such
as pointer to string or index) per token. Then, we would have something like this for the path `$.store.book[*].author`:

| Byte index  | 0              | 1        | 2        | 3        | 4        | 5-12             | 13-20           | 21-28 | 29-36             | 37-42   | 43-47  | 48-54    |
|-------------|----------------|----------|----------|----------|----------|------------------|-----------------|-------|-------------------|---------|--------|----------|
| Value       | 4              | 0        | 0        | 3        | 0        | 37               | 43              | 0     | 48                | “store” | “book” | “author” |
| Description | Length of path | Property | Property | Wildcard | Property | Pointer to store | Pointer to book | Empty | Pointer to author |         |        |          |

This example uses a 0-terminated, varint encoded length prefix or byte encoded length prefix for storing the string, but
this can change depending on what is easiest or fastest.
