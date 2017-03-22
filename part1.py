import tensorflow as tf

# node1 = tf.placeholder(tf.float32)
# node2 = tf.placeholder(tf.float32)  #  Implicito que usara float32
# node3 = node1 + node2
# node4 = 3 * node3

sess = tf.Session()

# print(sess.run(node4, {node1:5, node2:6}))

node5 = tf.Variable([.3], tf.float32)
node6 = tf.Variable([-.3], tf.float32)
node7 = tf.placeholder(tf.float32)

linear_model = node5 * node7 + node6

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {node7:[1,2,3,4]}))

correctValues = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - correctValues)

loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {node7:[1,2,3,4], correctValues:[0, -1, -2, -3]}))

fixNode5 = tf.assign(node5, [-1.])
fixNode6 = tf.assign(node6, [1.])
sess.run([fixNode5, fixNode6])

print(sess.run(loss, {node7:[1,2,3,4], correctValues:[0, -1, -2, -3]}))