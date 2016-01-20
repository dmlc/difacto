#ifndef _SARRAY_H_
#define _SARRAY_H_
#include "ps/sarray.h"
namespace difacto {

/**
 * \brief Shared array
 *
 * A smart array that retains shared ownership. It provides similar
 * functionalities comparing to std::vector, including data(), size(),
 * operator[], resize(), clear(). SArray can be easily constructed from
 * std::vector, such as
 *
 * \code
 * std::vector<int> a(10); SArray<int> b(a);  // copying
 * std::shared_ptr<std::vector<int>> c(new std::vector<int>(10));
 * SArray<int> d(c);  // only pointer copying
 * \endcode
 *
 * SArray is also like a C pointer when copying and assigning, namely
 * both copy are assign are passing by pointers. The memory will be release only
 * if there is no copy exists. It is also can be cast without memory copy, such as
 *
 * \code
 * SArray<int> a(10);
 * SArray<char> b(a);  // now b.size() = 10 * sizeof(int);
 * \endcode
 *
 * \tparam T the value type
 */
template <typename T>
using SArray = ps::SArray<T>;

}  // namespace difacto
#endif  // _SARRAY_H_
