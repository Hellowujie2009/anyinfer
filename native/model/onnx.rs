//! The small subset of ONNX IR 8 protobuf fields used by the V4 head.
//! Field numbers follow onnx/onnx.proto at ONNX v1.17.0.
pub const FLOAT: u64 = 1;
pub const INT64: u64 = 7;
pub const BFLOAT16: u64 = 16;

#[derive(Default)]
pub struct Message(Vec<u8>);

impl Message {
    fn varint(&mut self, mut n: u64) {
        while n >= 128 {
            self.0.push((n as u8 & 127) | 128);
            n >>= 7;
        }
        self.0.push(n as u8);
    }
    fn number(&mut self, field: u64, n: u64) {
        self.varint(field << 3);
        self.varint(n);
    }
    fn data(&mut self, field: u64, bytes: &[u8]) {
        self.varint((field << 3) | 2);
        self.varint(bytes.len() as u64);
        self.0.extend_from_slice(bytes);
    }
    fn string(&mut self, field: u64, s: &str) {
        self.data(field, s.as_bytes());
    }
    fn child(&mut self, field: u64, child: Message) {
        self.data(field, &child.0);
    }
}

fn tensor(name: &str, dtype: u64, dims: &[u64]) -> Message {
    let mut t = Message::default();
    for &dim in dims {
        t.number(1, dim);
    }
    t.number(2, dtype);
    t.string(8, name);
    t
}

fn value_info(name: &str, dtype: u64, dims: &[u64]) -> Message {
    let mut shape = Message::default();
    for &dim in dims {
        let mut dimension = Message::default();
        dimension.number(1, dim);
        shape.child(1, dimension);
    }
    let mut tensor_type = Message::default();
    tensor_type.number(1, dtype);
    tensor_type.child(2, shape);
    let mut value_type = Message::default();
    value_type.child(1, tensor_type);
    let mut info = Message::default();
    info.string(1, name);
    info.child(2, value_type);
    info
}

#[derive(Default)]
pub struct Graph(Message);

impl Graph {
    pub fn input(&mut self, name: &str, dtype: u64, dims: &[u64]) {
        self.0.child(11, value_info(name, dtype, dims));
    }
    pub fn output(&mut self, name: &str, dtype: u64, dims: &[u64]) {
        self.0.child(12, value_info(name, dtype, dims));
    }
    pub fn external(&mut self, name: &str, dtype: u64, dims: &[u64], filename: &str) {
        let mut t = tensor(name, dtype, dims);
        let mut location = Message::default();
        location.string(1, "location");
        location.string(2, filename);
        t.child(13, location);
        t.number(14, 1); // TensorProto.EXTERNAL
        self.0.child(5, t);
    }
    pub fn scalar(&mut self, name: &str, n: f32) {
        let mut t = tensor(name, FLOAT, &[]);
        t.data(9, &n.to_le_bytes());
        self.0.child(5, t);
    }
    pub fn integers(&mut self, name: &str, values: &[u64]) {
        let mut t = tensor(name, INT64, &[values.len() as u64]);
        let bytes: Vec<_> = values.iter().flat_map(|n| n.to_le_bytes()).collect();
        t.data(9, &bytes);
        self.0.child(5, t);
    }
    pub fn node(&mut self, op: &str, inputs: &[&str], output: &str, attr: Option<(&str, u64)>) {
        let mut n = Message::default();
        for input in inputs {
            n.string(1, input);
        }
        n.string(2, output);
        n.string(3, output);
        n.string(4, op);
        if let Some((name, value)) = attr {
            let mut a = Message::default();
            a.string(1, name);
            a.number(20, 2); // AttributeProto.INT
            a.number(3, value);
            n.child(5, a);
        }
        self.0.child(1, n);
    }
    pub fn finish(mut self) -> Vec<u8> {
        self.0.string(2, "DeepSeekV4SingleTokenHead");
        let mut opset = Message::default();
        opset.number(2, 13);
        let mut model = Message::default();
        model.number(1, 8);
        model.string(2, "anyinfer");
        model.child(7, self.0);
        model.child(8, opset);
        model.0
    }
}
