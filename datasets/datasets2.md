# Tensorflow tf.data API start (2)

본격적으로 **tf.data** 에 관하여 알아봅시다.

**tf.data**는 단순할 뿐 아니라 재사용이 가능하고 복잡한 입력 파이프 라인도 구축할 수 있습니다. 예를 들어 이미지 모델의 파이프 라인은 분산 파일 시스템의 파일에서 데이터를 가져온 후 각 이미지 데이터셋을 섞고 배치를 적용하는 것을 매우 직관적이고 쉽게 만들 수 있습니다.

**tf.data**의 특징은 다음과 같습니다.

- **tf.data.Dataset**는 각 요소가 하나 이상의 **tf.Tensor**를 포함하는 요소(elements)들을 가진다.
- **tf.data.Dataset**은 변환(transformation)을 실시 할 수 있고 변환(transformation)을 적용하면 변환된 **tf.data.Dataset**이 만들어진다.
- **tf.data.Iterator**는 데이터 집합에서 element 들을 추출하는 방법들을 제공합니다. element들을 주출할때 **Iterator.get_next()**을 실행하면 이전에 실행되었던 element의 다음 element를 반환한다. 그리고 input pipeline code와 model graph code 간에 interface역할을 한다.

---

## Basic mechanism

여기서는 **tf.data** 를 어떻게 사용하면 되는지 살펴봅시다. 이번 단락을 잘 이해하면 전체적으로 어떻게 pipeline을 만들면 되는지 감을 잡을 수 있을 것이라 기대합니다.

### 1. Create tf.data.Datasets

먼저 **tf.data.Datasets**을 만드는 방법에 대해 알아 봅시다. 디스크에 저장되어 있는 데이터들을 **tf.data.Datasets** 객체로 만들기 위해서는 `tf.data.Dataset.from_tensors()` 또는 `tf.data.Dataset.from_tensor_slice()` 를 사용합니다. 그리고 입력 데이터가 `TFRecode` 형태로 디스크에 저장되어 있다면 `tf.data.TFRecordDataset`를 사용합니다.

```python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)    # ==> tf.float32
print(dataset1.output_shapes)   # == > (10,)

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]),
     tf.random_uniform([4, 100], maxval=100, dtype=tf.int32))
)
print(dataset2.output_types)    # ==> (tf.float32, tf.int32)
print(dataset2.output_shapes)   # ==> ((), (100, ))

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)    # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)   # ==> (10, ((), (100, )))
```

```python
dataset = tf.data.Dataset.from_tensor_slices(
    {
        'a': tf.random_uniform([4]),
        'b': tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)
    }
)
print(dataset.output_type)      # ==> {'a' : tf.float32, 'b' : tf.int32}
print(dataset.output_shapes)    # ==> {'a' : () 'b' : (100, )}
```

### 2. Datasets transformation

**tf.data.Datasets** 객체가 만들어지면 메소드들을 호출하여 **tf.data.Datasets**을 여러 가지 형태로 변형할 수 있습니다. 예를 들어 각 요소(element) 별로도 변형이 가능 `(ex. tf.data.Dataset.map)` 하며 전체 데이터셋에 대해서도 변형할 수 있습니다. `(ex. tf.data.Dataset.batch)`. **tf.data.Dataset** 은 변형(transformation)과 관련된 많은 메소드들이 있으며 해당하는 메소드들의 리스트는 [여기](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)서 알아볼 수 있습니다.  

### 3. Create an tf.data.Iterator

Dataset 에서 input date에 대해 표현을 하면, **Iterator** 은 해 **Dataset** 세트의 요소에 엑세스하기 위해 작성합니다. **tf.data** API는 다음 반복자를 지원합니다.

- one-shot
- initializable
- reinitializable and
- feedable

**one-shot iterator**는 명시적으로 초기화 할 필요없이, Dataset 통해 한 번 반복하는 지원 반복자의 간단한 형태입니다. `원샷 반복자`는 기존 큐 기반 입력 파이프 라인이 지원하는 거의 모든 경우를 처리하지만 매개 변수화를 지원하지 않습니다.

```python
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


print(sess.run(next_element))   # ==> 0
print(sess.run(next_element))   # ==> 1
print(sess.run(next_element))   # ==> 2
print(sess.run(next_element))   # ==> 3
```

**initializable iterator**는 작업을 시작하기 전에 명시적으로 iterator.initializer를 실행하도록 요구합니다. 이 불편함을 감수하는 대신에 iterator를 초기화 할때 공급할 수 있는 하나 이상의 텐서를 사용하여 데이터 세트의 정의를 매개변수화(`tf.placeholder`) 할 수 있습니다. 아래 예제에서 차이점을 확인합니다.

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

**reinitializable iterator**는 여러가지를 초기화 할 수 있습니다. 예를 들어 일반화를 향상시키기 위해 입력 이미지의 랜덤으로 입력하는 train 을 위한 입력 파이프라인과 데이터가 얼마나 정확한지 확인하는 test 를 위한 입력 파이프 라인은 Dataset의 동일한 구조이지만 서로 다른 객체를 사용해야 됩니다. 이때 필요한 것이 `reinitializable` 입니다.

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