
Branch Protection Rules (set these up in GitHub repository settings):

1. main branch:
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Include administrators in restrictions

2. develop branch:
   - Require pull request reviews before merging
   - Require status checks to pass before merging

Feature Branch Workflow:
1. Create feature branch: git checkout -b feature/new-feature develop
2. Make changes and commit: git commit -am 'Add new feature'
3. Push changes: git push origin feature/new-feature
4. Create Pull Request to develop branch
5. After review and approval, merge to develop
6. Delete feature branch after merge

Release Process:
1. Create release branch: git checkout -b release/v1.0.0 develop
2. Version bump and final fixes
3. Merge to main: git checkout main && git merge release/v1.0.0
4. Tag release: git tag -a v1.0.0 -m 'Version 1.0.0'
5. Back-merge to develop: git checkout develop && git merge main
6. Delete release branch: git branch -d release/v1.0.0

