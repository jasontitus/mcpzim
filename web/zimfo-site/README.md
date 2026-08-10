# Zimfo public site

This static site supplies Zimfo's public marketing, privacy, support, and
TestFlight beta information pages. It deploys to the existing Firebase project
and Hosting site `tiltastech-zimfo`.

Deploy from this directory:

```sh
firebase deploy --only hosting:zimfo
```

Production URLs:

- Marketing: <https://tiltastech-zimfo.web.app/>
- Privacy: <https://tiltastech-zimfo.web.app/privacy>
- Support: <https://tiltastech-zimfo.web.app/support>
- Beta: <https://tiltastech-zimfo.web.app/beta>

Before public launch, verify that `support@tiltastech.com` is a working inbox
or alias and replace it throughout `public/` if a different support address is
preferred.
