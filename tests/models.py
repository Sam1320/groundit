"""Shared test models for groundit testing."""

from pydantic import BaseModel


# Simple test models
class Simple(BaseModel):
    name: str
    age: int


class Nested(BaseModel):
    profile: Simple
    active: bool


class WithLists(BaseModel):
    tags: list[str]
    profiles: list[Simple]


# Complex nested models from confidence tests
class Preferences(BaseModel):
    theme: str
    notifications: bool
    marketing_emails: bool


class Profile(BaseModel):
    name: str
    preferences: Preferences
    bio: str


class Stats(BaseModel):
    posts: int
    followers: int


class User(BaseModel):
    profile: Profile
    stats: Stats


class Metadata(BaseModel):
    created: str
    version: str


class NestedModel(BaseModel):
    user: User
    metadata: Metadata


TEST_OBJECT = NestedModel(
    user=User(
        profile=Profile(
            name="Alice",
            preferences=Preferences(
                theme="dark",
                notifications=True,
                marketing_emails=False
            ),
            bio="Alice is a software engineer at Google."
        ),
        stats=Stats(
            posts=42,
            followers=1337
        )
    ),
    metadata=Metadata(
        created="2024-01-01",
        version="1.0"
    )
) 