import User from "../models/User.js";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import nodemailer from "nodemailer";
import { nanoid } from "nanoid";

const register = async (req, res) => {
  try {
    const { email } = req.body;

    const existingEmail = await User.findOne({ email });
    if (existingEmail) {
      return res.status(400).json({ message: "The email is already exist!" });
    }

    const newUser = await User.create(req.body);

    newUser.password = undefined;

    return res
      .status(201)
      .json({ message: "Register successfully", user: newUser });
  } catch (error) {
    if (error.name === "ValidationError") {
      const validationError = {};

      for (let field in error.errors) {
        validationError[field] = error.errors[field].message;
      }

      return res
        .status(500)
        .json({ error: "Validation error", validationError });
    } else {
      console.error("Error at register: ", error);
      return res.status(500).json({ error: "Internal Server Error" });
    }
  }
};

const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) {
      return res
        .status(400)
        .json({ message: "Email or Password is Incorrect!" });
    }

    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) {
      return res
        .status(400)
        .json({ message: "Email or Password is Incorrect!" });
    }

    user.password = undefined;

    const accessToken = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET_KEY,
      { expiresIn: process.env.JWT_EXPIRES_TIME }
    );
    const refreshToken = jwt.sign(
      { userId: user._id },
      process.env.JWT_REFRESH_SECRET_KEY,
      { expiresIn: "7d" }
    );

    return res.status(200).json({
      message: "Login successfully",
      user: user,
      accessToken,
      refreshToken,
    });
  } catch (error) {
    console.error("Error during login: ", error);
    return res.status(500).json({ error: "Internal Server Error" });
  }
};

const refreshToken = async (req, res) => {
  const { token } = req.body;

  if (!token) {
    return res.status(403).json({ message: "Refresh token is required!" });
  }

  try {
    // Verify the refresh token
    const decoded = jwt.verify(token, process.env.JWT_REFRESH_SECRET_KEY);
    const userId = decoded.userId;

    // Generate a new access token
    const newAccessToken = jwt.sign({ userId }, process.env.JWT_SECRET_KEY, {
      expiresIn: process.env.JWT_EXPIRES_TIME,
    });

    return res.status(200).json({ accessToken: newAccessToken });
  } catch (error) {
    console.error("Refresh token error: ", error);
    return res.status(403).json({ message: "Invalid refresh token" });
  }
};

const sendResetMail = async (req, res) => {
  const { email } = req.body;
  const result = await User.findOne({ email });

  if (result) {
    const token = nanoid(32);
    var transport = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.NODE_USER,
        pass: process.env.NODE_PASS,
      },
    });

    const htmlBody = `Click here to <a href="http://localhost:5173/resetPassword?token=${token}">Reset Password</a>`;

    const info = await transport.sendMail({
      from: process.env.NODE_USER,
      to: email,
      subject: "Click Link To Reset Password",
      text: "Click Link To Reset Password",
      html: htmlBody,
    });

    await User.findOneAndUpdate({ email: email }, { verifytoken: token });
    return res.status(200).json({ message: "Email sent successfully." });
  } else {
    console.log("User does not exist: ");
    return res.status(400).json({ message: "Email could not be sent." });
  }
};

const resetPassword = async (req, res) => {
  const { password, token } = req.body;

  const salt = await bcrypt.genSalt(10);
  const hashedPassword = await bcrypt.hash(password, salt);

  try {
    await User.findOneAndUpdate(
      { verifytoken: token },
      { password: hashedPassword }
    );
    return res.status(200).json({ message: "Password changed successfully." });
  } catch (error) {
    return res
      .status(400)
      .json({ message: "Password change failed." });
  }
};

export { register, login, sendResetMail, resetPassword, refreshToken };
