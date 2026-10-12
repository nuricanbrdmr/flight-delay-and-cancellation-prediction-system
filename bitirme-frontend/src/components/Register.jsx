import { useState } from "react";
import { Form, Input, Button, Typography, Space, message } from "antd";
import {
  LockOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined,
  MailOutlined,
} from "@ant-design/icons";
import { axiosPrivate } from "../api/axios";

const { Title, Text, Link } = Typography;

const KAYIT_URL = "/auth/register";

// Regex kuralları
export const SIFRE_REGEX =
  /^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%]).{8,24}$/;

// Tooltip mesajları için özel bileşen
export const SifreDogrulama = () => (
  <div>
    <p>• Password must be 8-24 characters long</p>
    <p>• Must contain at least one uppercase and one lowercase letter</p>
    <p>• Must contain at least one digit and one special character (!@#$%)</p>
  </div>
);

const Register = () => {
  const [form] = Form.useForm();
  const [yukleniyor, setYukleniyor] = useState(false);
  const [basarili, setBasarili] = useState(false);

  const onFinish = async (values) => {
    setYukleniyor(true);
    try {
      await axiosPrivate.post(KAYIT_URL, {
        email: values.email,
        password: values.password,
      });
      setBasarili(true);
      message.success("Registration successful!");
      form.resetFields();
    } catch (err) {
      message.error("Registration failed");
      setYukleniyor(false);
    }
  };

  // Şifre eşleşme kontrolü
  const SifreEslestirme = (_, value) => {
    if (!value) {
      return Promise.reject("Please enter your password");
    }
    if (!SIFRE_REGEX.test(value)) {
      return Promise.reject(<SifreDogrulama />);
    }
    return Promise.resolve();
  };

  const EpostaKontrolu = (_, value) => {
    const emailRegExp = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!value || !emailRegExp.test(value)) {
      return Promise.reject("Enter a valid email address.");
    }
    return Promise.resolve();
  };

  if (basarili) {
    return (
      <div className="App">
        <Space
          direction="vertical"
          align="center"
          className="success-container"
        >
          <CheckCircleOutlined style={{ fontSize: 48, color: "#52c41a" }} />
          <Title level={2}>Registration Successful!</Title>
          <Link href="/login" strong>
            Log In
          </Link>
        </Space>
      </div>
    );
  }

  return (
    <div className="App">
      <Space
        direction="vertical"
        className="bg-white shadow-lg"
        style={{ width: "100%", maxWidth: 400, margin: "0 auto", padding: 24 }}
      >
        <Title level={2}>Sign Up</Title>

        <Form
          form={form}
          name="register"
          onFinish={onFinish}
          layout="vertical"
          scrollToFirstError
        >
          <Form.Item
            name="email"
            label="Email"
            tooltip={{
              title: "Please enter a valid email address.",
              icon: <InfoCircleOutlined />,
            }}
            rules={[{ validator: EpostaKontrolu }]}
          >
            <Input
              prefix={<MailOutlined />}
              placeholder="Enter a valid email address.."
            />
          </Form.Item>

          <Form.Item
            name="password"
            label="Password"
            tooltip={{
              title: <SifreDogrulama />,
              icon: <InfoCircleOutlined />,
            }}
            rules={[{ validator: SifreEslestirme }]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Enter a suitable password.."
            />
          </Form.Item>

          <Form.Item
            name="confirm"
            label="Confirm Password"
            dependencies={["password"]}
            rules={[
              { required: true, message: "Please enter your password again" },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue("password") === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject("Passwords do not match");
                },
              }),
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Confirm password.."
            />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" loading={yukleniyor} block>
              Sign Up
            </Button>
          </Form.Item>

          <Text>
            Already have an account? <Link href="/">Log In</Link>
          </Text>
        </Form>
      </Space>
    </div>
  );
};

export default Register;
