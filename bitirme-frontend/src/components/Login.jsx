import { useState } from "react";
import {
  Form,
  Input,
  Button,
  Alert,
  Typography,
  Space,
  Checkbox,
  message,
} from "antd";
import { MailOutlined, LockOutlined } from "@ant-design/icons";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { axiosPrivate } from "../api/axios";
import Cookies from "js-cookie";
import useAuth from "../hooks/useAuth";

const { Title } = Typography;
const LOGIN_URL = "/auth/login";

const Login = () => {
  const { setAuth, setPersist } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || "/";

  const [form] = Form.useForm();
  const [errMsg, setErrMsg] = useState("");

  const girisYap = async (values) => {
    try {
      const response = await axiosPrivate.post(LOGIN_URL, {
        email: values.email,
        password: values.sifre,
      });
      const accessToken = response?.data?.accessToken;
      const refreshToken = response?.data?.refreshToken;
      Cookies.set("refreshToken", refreshToken);
      setAuth({
        user: values.email,
        persist: values.persist,
        accessToken,
      });
      form.resetFields();
      message.success("Login successful.");
      navigate(from, { replace: true });
    } catch (err) {
      setErrMsg("Login failed");
    }
  };

  return (
    <div className="App">
      <Space
        direction="vertical"
        size={24}
        className="bg-white shadow-lg"
        style={{ width: "400px", padding: "32px" }}
      >
        <div style={{ textAlign: "center" }}>
          <Title level={2}>Log In</Title>
        </div>

        {errMsg && (
          <Alert
            message={errMsg}
            type="error"
            showIcon
            style={{ marginBottom: "5px", borderRadius: "8px" }}
          />
        )}

        <Form
          form={form}
          name="login"
          onFinish={girisYap}
          autoComplete="off"
          layout="vertical"
        >
          <Form.Item
            name="email"
            rules={[
              {
                required: true,
                message: "Please enter your email!",
              },
            ]}
          >
            <Input prefix={<MailOutlined />} placeholder="Email" size="large" />
          </Form.Item>

          <Form.Item
            name="sifre"
            rules={[
              {
                required: true,
                message: "Please enter your password!",
              },
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Password"
              size="large"
            />
          </Form.Item>

          <Form.Item name="persist" valuePropName="checked" className="!mb-4">
            <Checkbox onChange={() => setPersist((prev) => !prev)}>
              Keep me signed in
            </Checkbox>
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" block>
              LOG IN
            </Button>
          </Form.Item>

          <div className="text-center mb-3">
            <Link className="text-blue-500" to="/resetPassword?token=">
              Forgot password
            </Link>
          </div>

          <div style={{ textAlign: "center" }}>
            <p>
              Don't have an account?{" "}
              <Link className="text-blue-500" to="/register">
                Sign Up
              </Link>
            </p>
          </div>
        </Form>
      </Space>
    </div>
  );
};

export default Login;
