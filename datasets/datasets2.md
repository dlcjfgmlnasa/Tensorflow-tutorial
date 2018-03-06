# Tensorflow tf.data API start (2)

본격적으로 **tf.data** 에 관하여 알아봅시다.

**tf.data**는 단순할 뿐 아니라 재사용이 가능하고 복잡한 입력 파이프 라인도 구축할 수 있습니다. 예를 들어 이미지 모델의 파이프 라인은 분산 파일 시스템의 파일에서 데이터를 가져온 후 각 이미지 데이터셋을 섞고 배치를 적용하는 것을 매우 직관적이고 쉽게 만들 수 있습니다.

**tf.data**의 특징은 다음과 같습니다.

- **tf.data.Dataset**는 각 요소가 하나 이상의 **tf.Tensor**를 포함하는 요소(elements)들을 가집니다.
- **tf.data.Dataset**은 변환(transformation)을 실시 할 수 있고 변환(transformation)을 적용하면 변환된 **tf.data.Dataset**이 만들어집니다.
- **tf.data.Iterator**는 데이터 집합에서 element 들을 추출하는 방법들을 제공합니다. element들을 주출할때 **Iterator.get_next()** 을 실행하면 이전에 실행되었던 element의 다음 element를 반환합니다. input pipeline code와 model graph code 간에 interface역할을 하는 역할이라고 보시면 됩니다.

---

## Basic mechanism

여기서는 **tf.data** 를 어떻게 사용하면 되는지 살펴봅시다. 이번 단락을 잘 이해하면 전체적으로 어떻게 pipeline을 만들면 되는지 감을 잡을 수 있을 것이라 기대합니다.

### 1. Create tf.data.Datasets

먼저 **tf.data.Datasets**을 만드는 방법에 대해 알아 보도록 하겠습니다. 디스크에 저장되어 있는 데이터들을 **tf.data.Datasets** 객체로 만들어 주기 위해서는, `tf.data.Dataset.from_tensors()` 또는 `tf.data.Dataset.from_tensor_slice()` 를 이용하면 됩니다. 그리고 입력 데이터가 TFRecode 형태로 디스크에 저장되어 있으면 `tf.data.TFRecordDataset()`를 사용하시면 됩니다.

임의의 데이터를 생성하기 위해 `tf.random_uniform()` 를 이용하여 [4, 10] 형태의 정규분포를 가지는 matrix를 생성합니다.

```python
sample = tf.random_uniform([4, 10])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(sample))

"""
결과 :
[[ 0.90787792  0.61397159  0.18408167  0.7551167   0.73780155  0.71332455
   0.04578114  0.69684279  0.37828314  0.45624506]
 [ 0.00781178  0.34522772  0.33352196  0.59561861  0.11508143  0.03091383
   0.64267862  0.5854305   0.28963673  0.11909735]
 [ 0.42050052  0.19003737  0.81970799  0.50447953  0.08395636  0.72454298
   0.52941537  0.27786434  0.11810875  0.32241488]
 [ 0.64369309  0.82986867  0.90090048  0.17842996  0.41610706  0.27193487
   0.91392732  0.17527235  0.87324584  0.25085866]]
"""
```

임의로 생성한 데이터를 `tf.data.Dataset.from_tensor()`와  `tf.data.Dataset.from_tensor_slices()`에 넣어 봅시다.

```python
dataset1 = tf.data.Dataset.from_tensor_slices(sample)
dataset2 = tf.data.Dataset.from_tensor_slices(sample)
print(dataset1)                 # ==> <TensorDataset shapes: (4, 10), types: tf.float32>
print(dataset2)                 # ==> <TensorSliceDataset shapes: (10,), types: tf.float32>
```

결과 값을 확인해 보면 `tf.data.Dataset.from_tensor()`은 데이터의 전체를 저장하는것을 보실수가 있고  `tf.data.Dataset.from_tensor_slices()`은 전체데이터를 slice해서 저장하는것을 보실 수가 있습니다.

`tf.data.Dataset.from_tensor()` 또는 `tf.data.Dataset.from_tensor_slices()` 로 **tf.data.Dataset**객체가 만들어지면 객체안에 구성되는 element들은 동일한 구조로 구성되어 집니다. 각 element들은 `tf.Tensor` 형태이며 element 유형을 나타내는 `tf.DType`과 모양을 나타내는 `tf.TensorShape`로 구성되어져 있습니다.

`Dataset.output_types` 과 `Dataset.output_shape` 속성을 사용하면 **tf.data.Datset** 의 각 element들의 type과 shape를 확인 할 수 있습니다.

```python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)    # ==> tf.float32
print(dataset1.output_shapes)   # == > (10,)

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]),
     tf.random_uniform([4, 100], maxval=100, dtype=tf.int32))
)
print(dataset2.output_types)    # ==> (tf.float32, tf.int32)
print(dataset2.output_shapes)   # ==> (TensorShape([]), TensorShape([Dimension(100)]))

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)    # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)   # ==> (TensorShape([Dimension(10)]), (TensorShape([]), TensorShape([Dimension(100)])))

```

**tf.data.Dataset** 의 단일 요소에 collection.namedtuple 또는 dict를 이용하여 문자열을 탠서에 매핑할 하여 각 구성요소에 이름을 지정해 줄 수 있습니다. 아래 예제를 보시면 이해하시기 편하실

```python
# nametuples 를 이용한 구성요소 이름 지정
import collections
Sample = collections.namedtuple('sample_data', 'a b')
sample_data = Sample(
    tf.random_uniform([4]), tf.random_uniform([4, 100], maxval=100, dtype=tf.int32))
dataset = tf.data.Dataset.from_tensor_slices(sample_data)
print(dataset.output_types)     # ==> sample_data(a=tf.float32, b=tf.int32)
print(dataset.output_shapes)    # ==> sample_data(a=TensorShape([]), b=TensorShape([Dimension(100)]))
print(dataset.output_types.a)   # ==> <dtype: 'float32'>
print(dataset.output_types.b)   # ==> <dtype: 'int32'>
print(dataset.output_shapes.a)  # ==> ()
print(dataset.output_shapes.b)  # ==> (100, )


# dict 를 이용한 구성요소 이름 지정
dataset = tf.data.Dataset.from_tensor_slices(
    {
        'a': tf.random_uniform([4]),
        'b': tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)
    }
)
print(dataset.output_types)     # ==> {'a' : tf.float32, 'b' : tf.int32}
print(dataset.output_shapes)    # ==> {'a': TensorShape([]), 'b': TensorShape([Dimension(100)])}
print(dataset.output_types['a'])    # ==> <dtype: 'float32'>
print(dataset.output_types['b'])    # ==> <dtype: 'int32'>
print(dataset.output_shapes['a'])   # ==> ()
print(dataset.output_shapes['b'])   # ==> (100, )
```

### 2. Datasets transformation

**tf.data.Datasets** 객체가 만들어지면 메소드들을 호출하여 **tf.data.Datasets**을 여러가지형태로 변형을 할 수 있습니다. 예를들어 각 요소(element) 별로도 변형이 가능 `(ex. tf.data.Dataset.map())` 하고 전체 데이터셋에 대해서도 변형이 가능합니다. `(ex. tf.data.Dataset.batch())`. **tf.data.Dataset** 은 변형(transformation)과 관련된 많은 메소드들이 있는데 해당하는 메소드들의 리스트는 해당 링크를 확인하시면 됩니다.  [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

### 3. Create an tf.data.Iterator

이전 장에서 **tf.data.Datasets** 을 어떻게 생성하고 다루는지에 대해 알아 보았습니다. 이번 장에서는 **tf.data.Datasets** 의 element에 엑세스하기위한 **tf.data.Iterator**에 대해 알아 보도록 하겠습니다. 각 element에 대해 엑세스를 하여 실제 값을 받아 올 수 있어야 model에 넣어 학습이 가능 할 것입니다. **tf.data** 에서는 총 4가지 형태의 iterator를 제공합니다.

- one-shot
- initializable
- reinitializable and
- feedable

#### 1. one-shot Iterator

**one-shot iterator**는 명시적으로 초기화 할 필요없이 한 번만 반복 할 수 있는 가장 간단한 iterator의 형태입니다. **one-shot iterator** 는 기존 큐 기반 입력 파이프 라인이 지원하는 거의 모든 경우를 처리하지만 매개 변수화를 지원하지 않습니다.

> ※ 매개 변수화를 지원하지 않는다는 말은 다른 형태의 iterator를 설명을 보시면 이해가 빠르시리라 생각됩니다.

아래 예제를 보시면 `tf.data.Dataset.range(100)` 를 이용해 0~100까지 데이터를 가지는 **tf.data.Dataset**을 생성하고 `make_one_shot_iterator()`를 이용하여 iterator를 생성해주고 iterator로 부터 생성된 `get_next()` graph(next_elements) 를 실행하여 다음 element에 엑세스해 결과값을 출력합니다.

```python
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


print(sess.run(next_element))   # ==> 0
print(sess.run(next_element))   # ==> 1
print(sess.run(next_element))   # ==> 2
print(sess.run(next_element))   # ==> 3
```

`sess.run`를 할때마다 순차적으로 element들이 출력되는 것을 보실 수가 있습니다.

**one-shot iterator**는 한 번만 반복 할 수 있는 iterator라고 말씀드렸습니다. 아래 예제를 while문을 이용하여 두번 반복을 하였지만 실제로는 한번만 실행되는 것을 보실 수 있습니다.

```python
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

while True:
    try:
        print(sess.run(next_element), end=' ')  # ==> 0, 1, 2, 3, ..., 99
    except tf.errors.OutOfRangeError:   # 범위가 벗어나면 end를 출력하고 break
        print('end\n')
        break

while True:
    try:
        print(sess.run(next_element), end=' ') # 실행 안됨!!
    except tf.errors.OutOfRangeError:
        print('end\n')    # end 출력
        break
"""
결과 :
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 end

end
"""
```

만약 한번 더 반복시키고 싶으시다면 아래의 예제같이 iterator를 다시 만들어 주시면 됩니다.

```python
iterator2 = dataset.make_one_shot_iterator()
next_element2 = iterator2.get_next()
while True:
    try:
        print(sess.run(next_element2), end=' ')
    except tf.errors.OutOfRangeError:
        print('end')
        break
```

#### 2. initializable iterator

**initializable iterator**는 **one-shot iterator** 와 달리 작업을 시작하기 전에 명시적으로 `iterator.initializer`를 실행하도록 요구합니다. 이 불편함을 감수하는 대신에 iterator를 초기화 할때 공급할 수 있는 하나 이상의 텐서를 사용하여 데이터 세트의 정의를 매개변수화 할 수 있습니다 (제가 봐도 말이 어려운 거 같습니다. 아래 예제를 보시면 쉽게 이해 될꺼라 생각합니다.)

```python
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# dataset의 element의 갯수를 10개로 초기화 한다.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for _ in range(10):
    value = sess.run(next_element)
    print(value)               # ==> 0, 1, 2, 3, 4, .... , 9 (0부터 9까지)

# dataset의 element의 갯수를 100개로 초기화 한다.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for _ in range(100):
    value = sess.run(next_element)
    print(value)                # ==> 0, 1, 2, 3, 4, .... , 100 (0부터 100까지)
```

#### 3. reinitializable iterator

**reinitializable iterator**는 여러가지를 초기화 할 수 있습니다. 예를 들어 일반화를 향상시키기 위해 입력 이미지의 랜덤으로 입력하는 train 을 위한 pipeline 과 데이터가 얼마나 정확한지 확인하는 test 를 위한 pipline은 **tf.data.Dataset** 의 구조가 동일하지만 서로 다른 객체를 사용해야 됩니다. 이때 필요한 것이 `reinitializable` 쉽게 처리할 수 있습니다.

```python
# training과 validation datasets는 같은 구조를 가진다.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(100)

# reinitializable iterator는 structure에 의해서 정의 된다.
# training_dataset 과 validation_dataset의 output_types과 output_shapes
# 속성이 호환 된다.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# 20번을 반복하면서 train 과 validation 과정을 거친다.
for _ in range(20):
    # train dataset iterator를 초기화 한다.
    sess.run(training_init_op)
    for _ in range(100):
        print(sess.run(next_element))

    # validation dataset iterator를 초기화 한다.
    sess.run(validation_init_op)
    for _ in range(20):
        print(sess.run(next_element))
```

**feedable iterator**는 `tf.placeholder` 를 선택하기 위해 **Iterator** 각 호출에 사용하는 `tf.Session.run` 을 통해 이터레이터를 전환할때 데이터세트의 시작부분에서 반복기를 초기화 할 필요가 없습니다.

```python
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()

validation_dataset = tf.data.Dataset.range(50)

# feedable iterator는 handle placeholder 와 구조로 정의된다.
# training_dataset 과 validation_dataset의 output_types과 output_shapes
# 속성이 호환 될 수 있다.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# feedable 반복자는 다양한 종류의 반복자와 함께 사용가능하다.
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# Iterator.string_handle () 메소드는 handle placeholder를 제공하기 위해
# 평가되고 사용될 수있는 텐서를 리턴한다.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# 20번을 반복하면서 train 과 validation 과정을 거친다.
for _ in range(20):
    for _ in range(200):
        sess.run(next_element, feed_dict={handle: training_handle})

    # Run one pass over the validation dataset.
    sess.run(validation_iterator.initializer)
    for _ in range(50):
        print(sess.run(next_element, feed_dict={handle: validation_handle}))
```