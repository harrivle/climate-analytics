(If you know how to `git branch` and/or are familiar with `git` already, feel free to ignore these steps and work on your own branch or otherwise.)

### Some import notes:

* Don't use `git [COMMAND] --force` to force any commands.
* Don't use `git add *` unless you know what you're doing.
    * If you're not using `.gitignore`, you probably don't know what you're doing.
    * In fact, avoid using `*` in general unless you know what you're doing.

### When changes are ready to be pushed, go through these steps:

1. Use `git status` to check the status of your local repository.
   1. If you are behind `master`, attempt to `git pull` to update your local repository. If conflicts arise because of this, message in the group chat.
2. Use `git add [FILEPATH]` to add any new or modified files you wish to upload.
    1. If you make a mistake and want to remove a file from the commit, use `git reset [FILEPATH]` to unstage the file.
        * Alternatively, you can use `git restore --staged [FILEPATH]` for the same effect
3. Use `git commit -m "message"` with an appropriate short description of the commit in quotes.
    1. If you make a commit and want to undo the commit, you can use `git reset` withoout specifying a file to undo the commit.
4. Finally, use `git push` to upload your files to the GitHub repository.

### Miscellanneous notes:

* If you want to delete a file from the online repository (untrack the file) without deleting it locally, you can use `git rm [FILEPATH]` to do so.