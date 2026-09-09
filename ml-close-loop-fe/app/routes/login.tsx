import { useState } from "react";
import { useNavigate } from "react-router";
import InputSection from "~/components/Input-section";
import { Button } from "~/components/ui/button";
import { Field, FieldDescription, FieldGroup, FieldLegend, FieldSet } from "~/components/ui/field";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (username != "" || password != "") {
      sessionStorage.setItem("username", username);
      sessionStorage.setItem("password", password);
  
      navigate("/");
    }
  }

  return (
    <div className="flex flex-col w-full h-screen p-2 justify-center items-center">
      <form onSubmit={handleSubmit} className="pl-4 p-4 max-w-md w-full border rounded-lg">
        <FieldGroup>
          <FieldSet>
            <FieldLegend>Login</FieldLegend>
            <FieldDescription>
              Login with your username and password.
            </FieldDescription>
            <FieldGroup>
              <InputSection id="username" label="Username" onChange={(e) => setUsername(e.target.value)} required={true} />
              <InputSection id="password" label="Password" onChange={(e) => setPassword(e.target.value)} required={true} />
            </FieldGroup>
          </FieldSet>
          <Field>
            <Button type="submit">Login</Button>
          </Field>
        </FieldGroup>
      </form>
    </div>
  )
}