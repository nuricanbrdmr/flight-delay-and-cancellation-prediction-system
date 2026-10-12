import { Alert, Button, Form, Input, Space } from "antd";
import Typography from "antd/es/typography/Typography";
import {
  MailOutlined,
  LockOutlined,
  InfoCircleOutlined,
} from "@ant-design/icons";
import { SIFRE_REGEX, SifreDogrulama } from "./Register";
import { useSearchParams } from "react-router-dom";
import { useState } from "react";
import { axiosPrivate } from "../api/axios";

const { Title } = Typography;

const ResetPassword = () => {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");
  const [errMsg, setErrMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState();

  const handleSubmit = async (values) => {
    try {
      const response = await axiosPrivate.post("/auth/sendResetMail", {
        email: values.email,
      });
      setSuccessMsg(response.data.message);
    } catch (error) {
      setErrMsg("Password reset failed!");
    }
  };

  const handleReset = async (values) => {
    try {
      const response = await axiosPrivate.post("/auth/resetPassword", {
        password: values.password,
        token: token,
      });
      setSuccessMsg(response.data.message);
    } catch (error) {
      setErrMsg("Password reset failed!");
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

  return (
    <div className="App">
      <Space
        direction="vertical"
        size={24}
        className="bg-white shadow-lg"
        style={{ width: "400px", padding: "32px" }}
      >
        <div style={{ textAlign: "center" }}>
          <Title level={2}>
            {token ? "Set New Password" : "Forgot Password"}
          </Title>
        </div>

        {errMsg && (
          <Alert
            message={errMsg}
            type="error"
            showIcon
            style={{ marginBottom: "5px", borderRadius: "8px" }}
          />
        )}

        {successMsg && (
          <Alert
            message={successMsg}
            type="success"
            showIcon
            style={{ marginBottom: "5px", borderRadius: "8px" }}
          />
        )}

        <Form
          name="login"
          onFinish={!token ? handleSubmit : handleReset}
          autoComplete="off"
          layout="vertical"
        >
          {!token ? (
            <Form.Item name="email" rules={[{ validator: EpostaKontrolu }]}>
              <Input
                prefix={<MailOutlined />}
                placeholder="Email"
                size="large"
              />
            </Form.Item>
          ) : (
            <>
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
            </>
          )}

          <Form.Item>
            <Button type="primary" htmlType="submit" block>
              {token ? "RESET PASSWORD" : "SEND"}
            </Button>
          </Form.Item>
        </Form>
      </Space>
    </div>
  );
};

export default ResetPassword;
