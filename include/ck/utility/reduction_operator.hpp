// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type.hpp"

namespace ck {

namespace reduce {

// Every binary operator used in reduction is represented by a templated functor class. Each functor
// class must provide at least
// three members:
// 1) GetIdentityValue() -- the interface to return the "identity element" for the binary
// operator, "identity element" is the unique
//                    element in the algebraic space that doesn't affect the value of other elements
//                    when operated against them, and the concept is similar to zero vector in
//                    vector space
//                    (http://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/linearalgebra/VectorSpaces.pdf).
// 2) IsCompatibleInMemoryDataOperation() -- return true if the reduction task corresponding to this
// operator can use the InMemoryDataOperation to finalize, or else it return false
// 3) operator() -- the first argument of the operator must be both an input & output, and the
//                  corresponding variable usually stores
//                  the accumulated result of many operator() calls; the second argument is only an
//                  input. For indexable binary
//                  operator, the second version of operator() has third argument (which is an
//                  output) to indicate whether the
//                  accumulated value (the first argument) has changed, in which case the recorded
//                  accumulated index also need be
//                  changed.

struct Add
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::AtomicAdd ||
               operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value,
                      "The data type is not supported by the Add accumulator!");

        a = a + b;
    }
};

struct SquaredAdd
{
    template <class T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::AtomicAdd ||
               operation == InMemoryDataOperationEnum::Set;
    };

    template <class T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the SquaredAdd accumulator!");

        a = a + b * b;
    }
};

struct Mul
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(1.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value,
                      "The data type is not supported by the Mul accumulator!");

        a = a * b;
    }
};

struct Max
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return NumericLimits<T>::Lowest();
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_max to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

        if(a < b)
            a = b;
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }
};

struct Min
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return NumericLimits<T>::Max();
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_min to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Min accumulator!");

        if(a > b)
            a = b;
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Min accumulator!");

        if(a > b)
        {
            a       = b;
            changed = true;
        }
    }
};

struct AMax
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_max to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the AMax accumulator!");

        if(a < b)
            a = b;
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the AMax accumulator!");

        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }
};

struct fast_Add
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::AtomicAdd ||
               operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T,half_t>::value,
                      "The data type is not supported by the Add accumulator!");

	T c{1.0f};
	if(is_same<T,float>::value)
        {
	    asm volatile("\n \
		         v_fma_f32 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b), "0"(a));
        }
	else if(is_same<T,half_t>::value)
	{
	    asm volatile("\n \
		         v_fma_f16 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b),"0"(a));
        }
	else if(is_same<T,double>::value)
	{
            asm volatile("\n \
		         v_fma_f64 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b),"0"(a));
	}
	else
	{
	   a = a + b;
	}
    }
};

struct fast_Sub
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::AtomicAdd ||
               operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T,half_t>::value,
                      "The data type is not supported by the Add accumulator!");

        T c{-1.0f};
	if(is_same<T,float>::value)
        {
	    asm volatile("\n \
		         v_fma_f32 %0, %2, %1, %0\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b), "0"(a));
        }
	else if(is_same<T,half_t>::value)
	{
	    asm volatile("\n \
		         v_fma_f16 %0, %2, %1, %0\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b),"0"(a));
        }
	else if(is_same<T,double>::value)
	{
	    asm volatile("\n \
		         v_fma_f64 %0, %2, %1, %0\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b),"0"(a));
	}
	else
	{
	   a = a - b;
	}
    }
};

struct Add2
{
    template <typename T>
    __host__ __device__ static T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::AtomicAdd ||
               operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float2_t>::value || is_same<T, half2_t>::value,
                      "The data type is not supported by the Add accumulator!");
        T c{1.0f};
	if(is_same<T,float2_t>::value)
        {
	    asm volatile("\n \
		         v_pk_fma_f32 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b),
			     "0"(a));
        }
	else if(is_same<T,half2_t>::value)
	{
	    asm volatile("\n \
		         v_pk_fma_f16 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b),
			     "0"(a));
        }
    }
};

struct Sub2
{
    template <typename T>
    __host__ __device__ static T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::AtomicAdd ||
               operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float2_t>::value || is_same<T, half2_t>::value,
                      "The data type is not supported by the Add accumulator!");
        T c{-1.0f};
	if(is_same<T,float2_t>::value)
        {
	    asm volatile("\n \
		         v_pk_fma_f32 %0, %2, %1, %0\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b),
			     "0"(a));
        }
	else if(is_same<T,half2_t>::value)
	{
	    asm volatile("\n \
		         v_pk_fma_f16 %0, %2, %1, %0\n \
	                 "
	                   : "=v"(a)
	                   : "v"(c), "v"(b),
			     "0"(a));
        }
    }
};

struct Mul2
{
    template <typename T>
    __host__ __device__ static T GetIdentityValue()
    {
        return type_convert<T>(1.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float2_t>::value || is_same<T, half2_t>::value,
                      "The data type is not supported by the Mul accumulator!");
	if(is_same<T,float2_t>::value)
        {
	    asm volatile("\n \
		         v_pk_mul_f32 %0, %0, %1\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b),
	                     "0"(a));
	}
	else if(is_same<T,half_t>::value)
        {
	   asm volatile("\n \
		        v_pk_mul_f16 %0, %0, %1\n \
	                "
	                  : "=v"(a)
	                  : "v"(b),
	                    "0"(a));
	}
    }
};

struct fast_Max
{
    template <typename T>
    __host__ __device__ static T GetIdentityValue()
    {
        return NumericLimits<T>::Lowest();
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_max to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

	if(is_same<T,float>::value)
        {
	    asm volatile("\n \
		         v_max_f32 %0, %0, %1\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b),
	                     "0"(a));
	}
	else if(is_same<T,half_t>::value)
        {
	    asm volatile("\n \
		         v_max_f16 %0, %0, %1\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b),
	                     "0"(a));
	}
	else
	{
        if(a < b)
            a = b;
	}
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }
};

struct Max3
{
    template <typename T>
    __host__ __device__ static T GetIdentityValue()
    {
        return NumericLimits<T>::Lowest();
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_max to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline void operator()(T& a, T b, T c) const
    {
        static_assert(is_same<T, float>::value || is_same<T, half_t>::value ||
                          is_same<T, int32_t>::value,
                      "The data type is not supported by the Max accumulator!");

	if(is_same<T,float>::value)
        {
	    asm volatile("\n \
		         v_max3_f32 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b), "v"(c),
	                     "0"(a));
	}
	else if(is_same<T,half_t>::value)
        {
	    asm volatile("\n \
		         v_max3_f16 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b), "v"(c),
	                     "0"(a));
	}
	else
	{
	    asm volatile("\n \
		         v_max3_i32 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b), "v"(c),
	                     "0"(a));
	}
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }
};


struct fast_Min
{
    template <typename T>
    __host__ __device__ static T GetIdentityValue()
    {
        return NumericLimits<T>::Max();
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_max to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

	if(is_same<T,float>::value)
        {
	    asm volatile("\n \
		         v_min_f32 %0, %0, %1\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b),
	                     "0"(a));
	}
	else if(is_same<T,half_t>::value)
        {
	    asm volatile("\n \
		         v_min_f16 %0, %0, %1\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b),
	                     "0"(a));
	}
	else if(is_same<T,double>::value)
        {
	    asm volatile("\n \
		        v_min_f64 %0, %0, %1\n \
	                "
	                  : "=v"(a)
	                  : "v"(b),
	                    "0"(a));
	}
	else if(is_same<T,int32_t>::value)
        {
	    asm volatile("\n \
		        v_min_i32 %0, %0, %1\n \
	                "
	                  : "=v"(a)
	                  : "v"(b),
	                    "0"(a));
	}
	else
	{
        if(a < b)
            a = b;
	}
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }
};

struct Min3
{
    template <typename T>
    __host__ __device__ static T GetIdentityValue()
    {
        return NumericLimits<T>::Max();
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_max to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline void operator()(T& a, T b, T c) const
    {
        static_assert(is_same<T, float>::value || is_same<T, half_t>::value ||
                          is_same<T, int32_t>::value,
                      "The data type is not supported by the Max accumulator!");

	if(is_same<T,float>::value)
        {
	    asm volatile("\n \
		         v_min3_f32 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b), "v"(c),
	                     "0"(a));
	}
	else if(is_same<T,half_t>::value)
        {
	    asm volatile("\n \
		         v_min3_f16 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b), "v"(c),
	                     "0"(a));
	}
	else
	{
	    asm volatile("\n \
		         v_min3_i32 %0, %0, %1, %2\n \
	                 "
	                   : "=v"(a)
	                   : "v"(b), "v"(c),
	                     "0"(a));
	}
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }
};

template <typename T>
constexpr T GetIdentityValueForInMemoryDataOperation(InMemoryDataOperationEnum operation)
{
    T result = ck::type_convert<T>(0.0f);

    if(operation == InMemoryDataOperationEnum::AtomicMax)
        result = ck::NumericLimits<T>::Lowest();

    return (result);
};

template <InMemoryDataOperationEnum Operation, typename DataType>
struct InMemoryDataOperatonSupportedOnDataType
{
    static constexpr bool value = false;
};

template <typename DataType>
struct InMemoryDataOperatonSupportedOnDataType<InMemoryDataOperationEnum::AtomicAdd, DataType>
{
    static constexpr bool value =
        is_same<DataType, float>::value || is_same<DataType, double>::value;
};

template <typename DataType>
struct InMemoryDataOperatonSupportedOnDataType<InMemoryDataOperationEnum::AtomicMax, DataType>
{
    static constexpr bool value =
        is_same<DataType, float>::value || is_same<DataType, double>::value;
};

template <typename DataType>
struct InMemoryDataOperatonSupportedOnDataType<InMemoryDataOperationEnum::Set, DataType>
{
    static constexpr bool value =
        is_same<DataType, float>::value || is_same<DataType, double>::value ||
        is_same<DataType, half_t>::value || is_same<DataType, bhalf_t>::value ||
        is_same<DataType, int8_t>::value || is_same<DataType, int32_t>::value;
};

template <typename DataType>
struct InMemoryDataOperatonSupportedOnDataType<InMemoryDataOperationEnum::Add, DataType>
{
    static constexpr bool value =
        is_same<DataType, float>::value || is_same<DataType, double>::value ||
        is_same<DataType, half_t>::value || is_same<DataType, int8_t>::value ||
        is_same<DataType, int32_t>::value;
};


} // namespace reduce
} // namespace ck
